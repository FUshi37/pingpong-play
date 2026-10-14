# GPU0-QVEL/REAL-RMP V85 模型与实物等效部署说明

本文对应提交名 `GPU0-QVEL/REAL-RMP`。本次包含训练模型、可直接执行的
NumPy actor、57-D observation 构造、90 Hz 球状态保持、200 Hz 有状态
bounded-`q_ref` 以及 checkpoint fail-closed 校验。实物端可据此新增独立部署
profile；不得覆盖已有 GPU0-QVEL `simcode6_qvel67` 路径。

## 1. 发布模型与状态

- 课程：`goal_d455_measured_qvel_rmp_vertical_v85`，当前最新 GPU0 课程版本；
- 模型：
  `pingpong_controller/outputs/rl_sim/measured_qvel_rmp_vertical_v85_gpu0_seed20261004_20260828_stage23_failure_time_survival_online1/mjx_curriculum_best.pkl`；
- SHA-256：
  `0381dc68a8ea02b2b3db171cc7edcc45291ff1b510d944b5ae2cddd89505c93a`；
- 文件大小：`3,984,154 bytes`；
- stage：`rmp85_complete_nonexecution_full_episode_polish`，课程 Stage 24；
- stage/global update：`174 / 430`；true step：`2922905600`；
- seed / W&B：`20261004 / v72g0s7a1`；
- actor：`57 -> 256(tanh) -> 256(tanh) -> 7`，float32；
- critic：368-D asymmetric critic，仅训练使用；实物不能把 critic 输入拼入 actor。

选择 `best` 而不是 `last`：训练后的审计把 Stage-24 update 174 标记为当前最佳，
并为它生成了 1200/1200 步、16 hits 的确定性单回合回放。该 checkpoint 保存时
48-update rolling hits/length/full 分别为 `13.6399 / 0.91128 / 0.84181`，view
为 `0.99986`，绝对 apex/lift 为 `1.29423 / 0.16277 m`，counted-hit interval
为 `0.36190 s`，qvel/qacc exceedance 为 `0 / 0.00492`。后续 `last` 的 rolling
length 已下降到约 `0.90`，因此不发布尾部权重。

必须同时保留一个限制说明：V85 完成了 Stage 23，但 Stage 24 没有达到
`length >= 0.95`、`full >= 0.90` 的最终课程门槛，也没有进入后续大范围 RMP/PD
DR lessons。它是当前最佳、可供实物实现与分级验证的部署候选，不是已证明可直接
自主上机的硬件 release。

V84/V85 相对 V83 的改动只发生在训练奖励：从 Stage 23 起 V84 使用 5000
`m^-2` 低 apex 权重和 40-point/至少 13 hits 的 full event，V85 再加入
`-30 * unfinished_fraction` 的真实早停事件。actor 结构、观测、动作、`q_ref`、
RMP 边界和部署时序均未改变。

## 2. 部署参考代码

实物端首先参考：

- `tools/rl_2real/gpu0_qvel_real_rmp_reference.py`：发布模型路径/哈希、模型加载、
  exact actor、坐标变换、90 Hz scheduler、57-D observation、bounded-`q_ref`；
- `tools/rl_2real/mjx_policy_controller.py`：不依赖 JAX 的 checkpoint unpickler 和
  `NumpyMJXActor`；
- `tools/rl_sim/mjx_juggle_env.py`：训练源实现
  `integrate_bounded_measured_qvel_reference()`、`_make_obs()` 和
  `_point_to_base()`/`_vel_to_base()`；
- `tools/rl_sim/train_juggle_mjx_curriculum.py`：
  `_measured_qvel_rmp_vertical_v85_stages()`。

实物仓库以现有 `mjx_rl_policy.py`、`rl_policy.py`、`pingpong_node.py` 和
`interactive_policy_safety_probe.py` 的 simcode6 路径为接入点，只新增
`gpu0_qvel_real_rmp_v85` profile。

最小模型加载方式：

```python
from pingpong_controller.tools.rl_2real.gpu0_qvel_real_rmp_reference import (
    GPU0QvelRealRmpReference,
    load_released_actor,
)

actor = load_released_actor()  # 先验 SHA-256，后 unpickle，再校验 V85 payload
controller = GPU0QvelRealRmpReference(actor.mean_action)
```

加载器必须拒绝路径不存在、SHA 不符、非 V85 profile/stage/step、非 57/368/7
维、网络 shape 改变、teacher/residual actor、控制语义或关键环境字段改变的模型。
`.pkl` 只能在 SHA 校验通过后加载。完整 checkpoint 内含 optimizer/critic，部署时
只使用 `params["pi"]`。

actor forward 必须是：

```text
h1 = tanh(obs @ l1.w + l1.b)
h2 = tanh(h1  @ l2.w + l2.b)
mu = h2 @ out.w + out.b
action = clip(mu, -1, 1)
```

没有 observation normalization，没有 output tanh，没有随机 `log_std` 采样。
57 个全零 float32 输入的未裁剪 golden mean 为：

```text
[-0.013018085, 0.011029976, 0.019584768, 0.064141512,
  0.115628816,-0.122984409, 0.017757997]
```

## 3. 57-D actor observation

所有数组均为 float32、SI 单位，关节顺序固定为
`RightArm-0, ..., RightArm-6`。不得重排、标准化或加入 critic/DR 特征。

| index | 字段 | 单位与来源 |
| --- | --- | --- |
| 0:7 | measured `q` | encoder，rad |
| 7:14 | measured `dq` | encoder，rad/s |
| 14:17 | `base_q=[x,y,yaw]` | world m/m/rad；固定底座为 0 |
| 17:20 | `base_dq=[vx,vy,yaw_rate]` | world m/s,m/s,rad/s；固定底座为 0 |
| 20:23 | held `ball_pos_base` | base-local，m |
| 23:26 | held `ball_vel_base` | base-local，m/s |
| 26:29 | `racket_pos_base` | encoder FK 的 `right_ee_site`，m |
| 29:32 | `racket_vel_base` | 200 Hz FK 一阶差分，m/s |
| 32:35 | `ball_pos_base-racket_pos_base` | m |
| 35:42 | previous executed action | 上一 tick 已裁剪的 7-D action |
| 42:49 | `dq(t)-dq(t-1)` | 相邻 encoder tick，rad/s |
| 49 | `ball_age_norm` | `clip(age_s/0.5,0,1)` |
| 50:57 | `q_ref-q_measured` | 持久 reference error，rad |

reset、策略重连、控制权重新启用时，必须用同一帧新鲜 encoder/FK 数据原子设置：

```text
q_ref = q_measured
previous_dq = dq_measured
previous_action = zeros(7)
previous_racket_position = current_racket_position
```

首 tick 的 `dq` delta、previous action、`q_ref-q` 和 racket velocity 均为零。

## 4. 坐标系和 racket 速度

参考代码给出了 `world_point_to_base()` 和 `world_velocity_to_base()`。训练只使用
base 的 x/y/yaw，不使用 roll/pitch：

```text
dx = point_world.x - base_world.x
dy = point_world.y - base_world.y
point_base = [ cos(yaw)*dx + sin(yaw)*dy,
              -sin(yaw)*dx + cos(yaw)*dy,
               point_world.z ]

rel_vx = world_vx - base_vx + yaw_rate*dy
rel_vy = world_vy - base_vy - yaw_rate*dx
vel_base = [ cos(yaw)*rel_vx + sin(yaw)*rel_vy,
            -sin(yaw)*rel_vx + cos(yaw)*rel_vy,
             world_vz ]
```

注意 z 保留 XML/world z，不能套用历史报告中“减去 0.100 m base height”的显示
口径。racket world velocity 先按
`(right_ee_site(t)-right_ee_site(t-1))/0.005` 计算，再用上式变换。球和 racket
必须处于同一 base-local 坐标系；racket 状态来自 encoder FK，不来自视觉。

## 5. 90 Hz 球状态、时间戳和 age

actor 在 200 Hz 运行，球状态按 90 Hz fractional scheduler 刷新并在中间 tick
保持。刷新条件与训练一致：

```text
floor(step * 90/200) > floor(last_refresh_step * 90/200)
```

`FractionalBallObservation90Hz` 已实现该状态机。每次刷新同时原子更新 position、
velocity 和“这些数值所代表时刻”的 timestamp；未刷新时三者全部保持。

`age = control_time - represented_state_time`。它不是原始图像丢失时长、检测置信度、
predictor horizon 或上一帧到达时间。若 predictor 已把测量传播到当前 query time，
传播后的数值代表当前时刻，actor age 应接近零；不能保留原始曝光时间再重复计算延迟。
时间戳来自同一单调时钟，future timestamp 或 represented age 超过 `0.350 s` 必须
fail closed。

本 checkpoint 使用 `raw` 球速度模式，不在部署 adapter 再加 EMA/alpha-beta。
训练中的 refresh noise 为 `0.004375 m / 0.04375 m/s`；frame position bias
上限为 `[0.0025,0.0025,0.0025] m`，rotation bias 为
`[0.43125,0.43125,0.625] deg`，velocity bias 为
`[0.025,0.025,0.03625] m/s`，scale 为 `0.996125--1.003875`。这些是鲁棒性
分布，实物端不能人为注入。V85 没有训练相机 missing/dropout/reacquisition；
真实视觉失效应进入现有监督安全状态，不能输入全零球或假装继续可见。

## 6. 200 Hz action 与持久 bounded-`q_ref`

每 5 ms 使用同一 encoder snapshot 构造 observation、执行 actor 并更新一次：

| runtime 参数 | 值 |
| --- | --- |
| policy/control rate | 200 Hz，`dt=0.005 s` |
| action gain / scale multiplier | `1.0 / 1.0` |
| action filter / jerk limit | `0 / 0` |
| maximum joint feedback age | `0.010 s` |
| ball observation rate | 90 Hz fractional |
| ball age clip / invalid timeout | `0.5 / 0.350 s` |
| output | 7-D rad position，`MechUnitCmd.jnt_pos` |

```text
action      = clip(actor_mean(obs), -1, 1)
desired_dq  = action * velocity_limit_rad_s
q_integral  = q_ref + desired_dq * 0.005
error_max   = velocity_limit_rad_s * horizon_s
lower       = max(training_joint_low,  q_measured - error_max)
upper       = min(training_joint_high, q_measured + error_max)
q_ref_next  = clip(q_integral, lower, upper)
```

随后保存 `previous_action=action`、`previous_dq=dq_measured` 和
`q_ref=q_ref_next`。不得从 `q_measured` 每 tick 重新初始化 `q_ref`，不得使用旧
`q_cmd/dq_cmd`，不得再执行 acceleration integrator、action LPF、jerk filter、
anti-windup 或输出 delay。

| joint | q low/high (rad) | velocity (deg/s) | acceleration envelope (deg/s²) | horizon (ms) | max `q_ref-q` (deg) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | `-2.0944 / 3.14159` | 210 | 1300 | 18 | 3.78 |
| 1 | `-2.96706 / 0.15708` | 210 | 1300 | 19 | 3.99 |
| 2 | `-3.05433 / 3.05433` | 240 | 1800 | 19 | 4.56 |
| 3 | `-0.174533 / 2.25147` | 240 | 3000 | 18 | 4.32 |
| 4 | `-3.05433 / 3.05433` | 300 | 3000 | 18 | 5.40 |
| 5 | `-1.65806 / 1.65806` | 300 | 3000 | 18 | 5.40 |
| 6 | `-1.5708 / 1.5708` | 300 | 3000 | 18 | 5.40 |

训练/启动姿态，按上述关节顺序，单位 deg：

```text
[5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214]
```

外层直接 QVEL path 不使用 acceleration clip；表中的 acceleration 是训练 plant/
安全评估包络，不能在 actor 和 `q_ref` 之间擅自增加第二个积分器。实物已有安全
limiter 和硬限位仍保留；如果它们持续改变 `q_ref_next`，这是部署不等效/工作区不匹配，
必须记录并停止放权。

## 7. 机器人 RMP 边界

`q_ref_next` 是唯一策略输出边界。经过现有独立安全检查后，把它作为 7-D rad
位置目标写入机器人 RMP 输入 `MechUnitCmd.jnt_pos`，正常按 200 Hz 发布：

```text
encoder q/dq + ball + encoder FK
  -> 57-D actor
  -> persistent bounded q_ref
  -> existing safety limiter
  -> MechUnitCmd.jnt_pos
  -> robot existing RMP / low-level controller
```

训练中的 recovered RMP 是机器人已有 RMP 的 simulator surrogate。禁止在策略进程
复制 recovered RMP、`recovered_rmp_rmpmd_v2` 的 XML PD、1 kHz 五个 surrogate
substeps、18/19 ms RMP output delay、actuator plant、inverse compensation、第二个
planner 或 post-RMP hold。也禁止把 simulator `command_q` 当作 actor 输出。

训练分布含 target-publication hold `0.08`，其中 `0.022` 为 2--9 tick tail；实物
正常逐 tick 发布，不主动随机漏发。真实漏发时机器人 RMP 自然保持上一目标即可，
不能在策略层再叠加 hold 状态机。

## 8. 相比 GPU0-QVEL（simcode6）的部署变化

对比基线模型提交 `e012d07400a99e95b50c2dcf7200d6ede312dfa6`，companion
`d269517c4e79cc53b9024dc6bdf9eff496a653d3`：

| 项目 | simcode6 | GPU0-QVEL/REAL-RMP V85 |
| --- | --- | --- |
| actor observation | 67-D delay-conditioned | 57-D measured-feedback |
| 反馈/积分状态 | `q_cmd/dq_cmd` | encoder `q/dq` + persistent `q_ref` |
| 后缀 | command/delay features | `dq(t)-dq(t-1)` + age + `q_ref-q` |
| QVEL scale | 0.85 | 1.0 |
| action filter / slew | 15 ms / 30 s^-1 | 0 / 0 |
| ball refresh | 60 Hz | 90 Hz fractional |
| held-frame age | 历史路径强制 0 | represented-state age / 0.5 |
| target | command-state integrated q | measured bounded persistent `q_ref` |
| simulation plant | fitted delayed actuator | recovered RMP + XML PD |
| real boundary | drive position target | robot existing RMP position input |

这不是“只换模型”。旧 67-D adapter 即使能读 checkpoint，也不能生成等效 observation
和有状态输出，必须新增独立 runtime profile。

## 9. 实物端最小实现与验收

1. 使用发布路径加载模型，先验证 SHA，再调用 V85 payload validator；验证全零输入
   golden action。
2. 新增独立 57-D profile；固定关节顺序、float32、SI 单位和 actor 字段 index。
3. 用同一 encoder snapshot 计算 q/dq、FK racket、`dq` delta 和 `q_ref-q`；reset
   四个持久状态必须原子完成。
4. 正确实现 world/base 变换与 90 Hz held ball state；逐 tick记录 represented
   timestamp、age 和 refresh 标记。
5. 先做固定 observation 的 actor parity，再做 recorded q/dq/ball trace 的逐 tick
   observation/action/`q_ref` parity；跨 NumPy/JAX float32 实现容差使用
   `atol <= 5e-6`。
6. `q_ref_next` 只进入已有 safety limiter 和机器人 RMP；确认没有重复 RMP、PD、
   delay、filter、planner 或 actuator model。
7. 依次进行不发布 shadow、无球低增益/限空间、监督有球测试。记录 raw/clipped
   action、57-D obs、q/dq、q_ref、safe target、发布/encoder/ball timestamp、RMP
   feedback、安全干预、球状态和 racket FK。

任何非有限值、关节反馈年龄大于 `10 ms`、球 future timestamp、球 represented age
大于 `350 ms`、关节顺序/模型哈希不匹配都必须在 actor 调用或发布前 fail closed。
相机分布、新球 outcome 和正式 RMP/PD pointwise coverage 仍未完成；频繁 safety
limiter 干预是契约失败，不是正常部署现象。

## 10. 仿真训练与验证 companion 代码

模型、部署 adapter 和部署合约已经由 `GPU0-QVEL/REAL-RMP` 模型提交
`bd75709` 发布。本节对应独立的仿真 companion 提交：实物端可用它核对训练时
的 actor 输入、策略积分、RMP surrogate、物理参数、DR 和验证环境，但不能把全部
仿真执行栈原样串到机器人已有 RMP 之前。

### 10.1 精确训练入口与调用链

V85 的唯一正式入口是：

```text
run_gpu0_measured_qvel_rmp_vertical_v85_cgroup_resume_v83_stage23_update100.sh
  -> run_gpu0_measured_qvel_rmp_vertical_v85_resume_v83_stage23_update100.sh
  -> run_gpu0_measured_qvel_rmp_vertical_v37_resume_v36.sh
  -> run_with_host_memory_guard.sh
  -> train_juggle_mjx_curriculum.py
```

入口固定 `goal_d455_measured_qvel_rmp_vertical_v85`、GPU0、57-D actor、368-D
critic、1024 environments、128 rollout steps、minibatch 16384、2 epochs、
LR `5e-5`、clip `0.10`、target KL `0.004`，并从 V83 Stage-23 update 100 的
指定 SHA checkpoint 恢复。不要从脚本名猜测配置；最终等效参数以已发布 V85
checkpoint 的 `env_config`、本说明和 profile builder 三者一致为准。

V85 是 continuation-only。启动器还要求本机 V83 源 checkpoint、RMP DataTracer
回放报告和 bounded-`q_ref` screen JSON；这些实验输入及中间 checkpoint 不属于
GitHub companion 提交。因此该提交足以审计已发布模型并指导实物实现，但不承诺
仅靠 GitHub 文件逐 update 重放 V83 到 V85 的优化历史。

### 10.2 仿真源码职责

| 文件 | 实物端参考内容 |
| --- | --- |
| `mjx_juggle_env.py` | 57-D observation、90 Hz fractional refresh、bounded-`q_ref`、reward、RMP/PD plant 与 DR |
| `recovered_rmp_jax.py` | 训练本地 recovered-RMP surrogate；只用于仿真对齐，不部署到实物策略进程 |
| `delay_control.py` | 共享延迟/hold 控制原语 |
| `rmp_training_evidence.py` | 启动前 RMP 回放与 DR 证据的 fail-closed 校验 |
| `train_juggle_mjx_ppo.py` | actor/critic、PPO、GAE、checkpoint payload 和策略前向 |
| `train_juggle_mjx_curriculum.py` | V85 完整课程、Stage 23/24 reward 以及 resume payload 校验 |
| `validate_juggle_mjx_ppo.py` | 从 checkpoint 恢复环境并进行 MJX 回放验证 |
| `screen_juggle_mjx_checkpoints.py` | 同环境条件下批量筛选同 shape checkpoint |
| `screen_bounded_qref_horizons.py` | 逐关节 bounded-`q_ref` horizon 的仿真筛选 |
| `diagnose_measured_qvel_rmp_path.py` | measured-QVEL/RMP 权限路径和首拍 credit 诊断 |
| `run_with_host_memory_guard.sh` | 训练进程主机内存安全停止，不属于策略推理 |

基础 `moz1_pd.xml`、`camera_calibration.py`、`mjx_smoke.py`、
`rl_juggle_env_random.py` 和 `sim2real_bridger.py` 已在仓库历史中存在且本次没有
修改；它们仍是运行上述代码的依赖，不需要为这次 companion 重复制造文件变更。

共享 trainer 同时保留历史 profile、后续 V86 和相邻 GPU1 实验，这是保持旧
checkpoint 可加载所必需的兼容结构。部署 V85 时必须显式选择 V85 profile 并通过
已发布 adapter 的 checkpoint validator，不能把文件中编号最大或默认 profile 当成
本模型配置。本次只提交 V85 的 launcher，不提交 V86 launcher、GPU1 launcher 或
其他实验脚本。

### 10.3 实物端哪些内容照搬、哪些不能照搬

应逐项等效实现：57-D actor 顺序、float32/SI 单位、200 Hz actor、90 Hz held ball
state、represented-state age、persistent bounded-`q_ref` 状态与 reset、关节限位和
输出到机器人现有 RMP 的位置目标边界。

只用于训练、不得进入实物 actor path：368-D critic、reward/gate、DR 随机采样、
recovered RMP 的 1 kHz 子步、XML PD、RMP output delay、仿真 contact/ball dynamics、
PPO/GAE、W&B 和训练内存保护。真实机器人正常运行时也不要主动注入 observation
noise、frame bias、target-publication hold 或漏帧；这些是训练鲁棒性分布，不是部署
前处理。

## 11. 两次提交的边界

已完成的模型提交 `bd75709` 包含：

```text
.../mjx_curriculum_best.pkl
tools/rl_2real/gpu0_qvel_real_rmp_reference.py
tools/rl_2real/GPU0_QVEL_REAL_RMP_GUIDE.md
test/test_gpu0_qvel_real_rmp_reference.py
```

新的仿真 companion 只增加本节列出的训练/验证源码、V85 启动链、课程注册说明和
本说明文档的更新。不重复提交模型，不提交 `last`、archive、源恢复 checkpoint、
progress CSV、RMP evidence 输出、W&B、视频、分析报告、`__pycache__`、V86/GPU1
launcher 或无关测试。

## 12. 仿真 companion 提交指令

工作区包含大量并行实验，禁止使用 `git add .`。在仓库根目录执行：

```bash
git add -- \
  TRAINING_CURRICULA.md \
  pingpong_controller/tools/rl_2real/GPU0_QVEL_REAL_RMP_GUIDE.md \
  pingpong_controller/tools/rl_sim/delay_control.py \
  pingpong_controller/tools/rl_sim/mjx_juggle_env.py \
  pingpong_controller/tools/rl_sim/recovered_rmp_jax.py \
  pingpong_controller/tools/rl_sim/rmp_training_evidence.py \
  pingpong_controller/tools/rl_sim/train_juggle_mjx_ppo.py \
  pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py \
  pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py \
  pingpong_controller/tools/rl_sim/screen_juggle_mjx_checkpoints.py \
  pingpong_controller/tools/rl_sim/screen_bounded_qref_horizons.py \
  pingpong_controller/tools/rl_sim/diagnose_measured_qvel_rmp_path.py \
  pingpong_controller/tools/rl_sim/run_with_host_memory_guard.sh \
  pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v37_resume_v36.sh \
  pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v85_resume_v83_stage23_update100.sh \
  pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v85_cgroup_resume_v83_stage23_update100.sh

git commit -m "sim: add GPU0 QVEL V85 recovered-RMP training stack"
```
