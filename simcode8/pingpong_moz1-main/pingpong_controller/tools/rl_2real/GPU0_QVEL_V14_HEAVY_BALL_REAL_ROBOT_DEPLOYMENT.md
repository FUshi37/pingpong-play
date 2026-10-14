# GPU0 QVEL V14 重球策略实物部署与提交说明

## 1. 发布模型

- Git 基线：`69b9bf34`（`sim: version GPU1 sport actuator training stack`）。
- 训练 profile：`goal_d455_sport_taskspace_qvel_vertical_v14`。
- W&B：`fushi37/pingpong-mjx/yjqfl80b`。
- 发布 checkpoint：
  `pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/mjx_curriculum_last.pkl`。
- SHA-256：
  `2ca715d71fe19a0058b8cac710574630bdff580ec28157ac80a3e1d70f8cef47`。
- 网络：actor `67D -> 256 -> 256 -> 7D`；asymmetric critic 231D，仅训练使用。
- 完成位置：stage 24/24
  `qvel_v14_heavy_ball_3p7g_lower_elasticity_target`，stage update 373，
  global update 468，global step `3457417216`。

这里发布 `last` 而不是训练时写入的 `best`。两者使用 seed `20260812`、
固定 3.7 g 球、`solref` time `0.005 s`、damping `0.90`、64 个确定性 episode
做了同条件复核：

| checkpoint | hits | mean steps | full rate | view | mean hit-vxy | failure |
|---|---:|---:|---:|---:|---:|---|
| `best`, target update 114 | 17.05 | 1186.1 | 0.9844 | 1.000 | 0.063 m/s | 1 `ball_too_low` |
| `last`, target update 373 | 16.80 | 1200.0 | 1.0000 | 1.000 | 0.061 m/s | none |

`best` 的平均击球略高，但有一局提前失败；`last` 同时具备课程正式收敛证书和
64/64 满时域结果，因此用于实物部署。

## 2. 收敛与回放证据

最终 32-update 训练窗口：

| metric | value |
|---|---:|
| mean hits | 15.819 |
| mean length fraction | 0.9951 |
| full-episode rate | 0.9855 |
| ball view in bounds | 0.9979 |
| mean / RMS hit-vxy | 0.0754 / 0.0887 m/s |

严格 final self-probe 在 128/128 episodes 上达到 length/view/full rate 均为
`1.0`，16.38 hits、RMS hit-vxy `0.0745 m/s`。发布 checkpoint 的独立固定重球
确认是 64/64 满 1200 步、16.80 hits，只有正常 horizon truncation。

最终视频位于 `final_video/gpu0_v14_final_heavy_3p7g_60hz.mp4`：1200/1200
步、19 hits、view/camera/hit-band rate 均为 `1.0`、mean hit-vxy
`0.057 m/s`。部署提交只保留视频、action plot、精简 validation CSV 和 README；
体积较大的原始 action/observation trace 留在本地实验归档。

## 3. 训练配置

### 3.1 PPO 与课程

| 配置 | 值 |
|---|---:|
| GPU / seed | GPU0 / 61010 |
| environments / rollout | 1024 / 128 steps |
| minibatch / epochs | 16384 / 4 |
| learning rate | `2e-5` |
| gamma / GAE lambda | `0.9995` / `0.99` |
| PPO clip / target KL | `0.12` / `0.008` |
| log-std bounds | `[-3.2, -2.5]` |
| entropy coefficient | `0.0006` |
| actor-anchor KL coefficient | `0.015` |
| failure focus | hit `<4`, weight `1.5`, tail 160 |
| convergence window | 32 updates, at least 64 episodes |
| stage update cap | none |
| final blocking validation | 128 envs, 1200 steps, length/view `1.0` |

V14 从 24-update 重球 bridge checkpoint 继续，先覆盖 2.90--3.70 g、damping
0.66--1.06，再进入最终 3.45--3.95 g、damping 0.72--1.08。ball contact
time range 保持 0.002--0.008 s；没有用无效的 record_new3 sim mirror 作为标签。

### 3.2 策略动作契约

这是本次与旧 GPU0/GPU1 checkpoint 最重要的区别：

```text
7D actor output [-1, 1]
  -> normalized-action LPF (15 ms) and slew limit (30 / s)
  -> joint velocity target = action * qvel_limit * 0.85
  -> physical joint-acceleration envelope
  -> integrate once to a q-only position target
  -> real drive
```

checkpoint 的 `action_command_mode=velocity`、`action_velocity_scale=0.85`。
不得沿用旧控制器的 acceleration double-integrator，否则同一个 actor 输出会被多积分
一次。实物控制器现在从 checkpoint 读取动作模式；旧 checkpoint 缺少该字段时仍默认
`acceleration`，保持向后兼容。

右臂限制：

| joint | velocity (deg/s) | acceleration (deg/s²) |
|---|---:|---:|
| RightArm-0 | 210 | 1300 |
| RightArm-1 | 210 | 1300 |
| RightArm-2 | 240 | 1800 |
| RightArm-3 | 240 | 3000 |
| RightArm-4 | 300 | 3000 |
| RightArm-5 | 300 | 3000 |
| RightArm-6 | 300 | 3000 |

训练/启动右臂姿态（deg，RightArm-0..6）：

```text
5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214
```

控制器和 ROS init-pose 判定都从 checkpoint 恢复这组值，不再把旧全局
`TARGET_DEGREES` 当作该策略的右臂启动姿态。

### 3.3 观测与时序契约

| 配置 | 值 |
|---|---:|
| policy/control rate | 200 Hz (`dt=0.005 s`) |
| camera/ball observation rate | 60 Hz fractional refresh |
| base observation | 50D |
| delay-conditioned extension | 17D |
| total actor observation | 67D |
| position refresh noise | 0.002 m |
| velocity refresh noise | 0.07 m/s |
| observation latency DR | 0--2 control steps |
| action latency DR | 0 steps |
| ball observation age clip | 0.5 s |
| velocity observer | `raw` |
| fixed scalar delay feature | 45 ms / 9 control steps |

17D extension = normalized delay 1D + commanded qvel 7D + active-command error
7D + contact phase 2D。相机新帧之间必须保持上次有效球位置/速度，并累加 age；
丢球时传 `ball_valid=False`，不能伪造零球状态。ROS 节点把图像时间戳传给控制器，
避免 200 Hz timer 对同一 60 Hz 图像重复更新观测器。

实物运行必须显式设置 `rl_ball_obs_age_clip:=0.5`，因为节点历史默认值是 0.2 s。

### 3.4 执行器与控制栈

checkpoint 固定：

```text
actuator_cmd_model = second_order
actuator_compensation_mode = none
arm_servo_target_tracking_planner = false
enable_delay_conditioning = true
right_arm_pd_profile = sport_taskspace_fit_v1   # simulation only
```

标称二阶执行器参数：

| joint | wn (rad/s) | zeta | gain | delay (ms) |
|---|---:|---:|---:|---:|
| RightArm-0 | 20.363849 | 0.391768 | 0.997884 | 45 |
| RightArm-1 | 22.655975 | 0.366169 | 0.996612 | 50 |
| RightArm-2 | 22.650471 | 0.345738 | 0.992312 | 45 |
| RightArm-3 | 21.730533 | 0.346000 | 0.990942 | 40 |
| RightArm-4 | 19.663364 | 0.347896 | 0.982327 | 35 |
| RightArm-5 | 22.815526 | 0.345814 | 0.992695 | 45 |
| RightArm-6 | 23.645817 | 0.380713 | 0.983194 | 55 |

episode DR：wn/zeta scale 0.90--1.10、gain scale 0.99--1.01、delay offset
-2--+1 control steps。

这些参数描述仿真里的真实执行器替身。实物本身已经是被建模对象，所以部署端不要
再加入 second-order filter、35--55 ms software delay、inverse MPC、servo planner
或额外 actuator governor。`predict()` 保留 command history 以构造 67D 观测，但
直接发布当前 q-only target，再交给独立 `RightArmCommandSafetyLimiter` 和硬件急停链路。

因此两侧实际对应关系是：

```text
simulation: q-only target -> per-joint 35--55 ms delay -> fitted second-order model
            -> sport_taskspace_fit_v1 PD -> MuJoCo joint
real:       q-only target -> safety limiter -> real sport-mode drive -> physical joint
```

训练配置中 `actuator_cmd_filter=true`、`actuator_cmd_model=second_order`、
`actuator_delay_observation_only=false`，所以二阶模型及逐关节延迟确实作用到了仿真 plant，
并非只作为 observation feature。部署端不重复执行正向模型，是为了让“仿真模型”对应
“真实执行器”，而不是让“仿真模型”对应“软件模型 + 真实执行器”的串联双 plant。

这一结构只能在执行器辨识有效时缩小 sim-real gap，不能凭配置本身保证一致。首次实物
放权前必须用与辨识时相同的 sport-mode/固件/内环配置，记录 200 Hz 的发送位置目标、
实测 q/dq 和时间戳，并复核各关节延迟、增益、自然频率与阻尼是否落在训练 DR 包络内：
wn/zeta 0.90--1.10、gain 0.99--1.01、delay 标称值 -10--+5 ms。超出包络时应先重新
辨识并更新仿真 plant/DR，再训练或确认 checkpoint；不应在部署端盲加一层同名滤波器。

`sport_taskspace_fit_v1` 的 MuJoCo kp/kv 是仿真 composite-plant 拟合值，不能写入
实物驱动器私有 PD。`models/moz1_pd.xml` 的本地实验修改不属于部署提交。

## 4. 代码变更

- `delay_control.integrate_action_command()` 实现与 MJX 一致的 velocity/acceleration
  双模式动作语义、qvel/qacc 限制和旧模型兼容。
- `MJXPolicyController` 从 checkpoint 恢复动作模式、velocity scale 和训练启动姿态，
  summary 明确打印这些契约。
- `pingpong_node.py` 使用 checkpoint 右臂姿态判定 init pose，并把相机时间戳传入
  controller；节点日志显示动作模式。
- `ball_velocity_observer.py` 对相同相机时间戳只更新一次；本 checkpoint 使用 raw mode。

训练 companion code 包括 qvel action environment、V12/V13/V14 课程、重球验证参数、
host-memory guard。基础 XML 本身保持提交版本；训练时由 profile 生成临时
sport PD XML。

## 5. 部署前 smoke test

```bash
cd /home/yangzhe/Project/pingpong_controller

PYTHONPATH=/home/yangzhe/Project/pingpong_controller \
/home/yangzhe/miniconda3/envs/pingpong/bin/python \
  pingpong_controller/tools/rl_2real/mjx_policy_controller.py \
  --checkpoint pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/mjx_curriculum_last.pkl \
  --robot-xml pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --require-fk --steps 2 --ball-valid \
  --ball-pos 0.02,-0.35,1.20 --ball-vel 0,0,-0.3 \
  --arm-q-deg 5.736,-44.399,30.683,97.142,49.323,-12.269,14.214 \
  --arm-dq-deg-s 0,0,0,0,0,0,0
```

summary 必须包含：

```text
step: 3457417216
obs_dim: 67
act_dim: 7
action_command_mode: velocity
action_velocity_scale: 0.85
delay_conditioning: True
delay_extra_dim: 17
tau_act_s: 0.045
actuator_compensation_mode: none
drive_target_tracking_planner: False
fk_enabled: True
```

## 6. ROS2 实物配置

首次只验证初始化与安全链时把 `rl_action_gain` 设为 0：

```bash
cd /home/yangzhe/Project/pingpong_controller
colcon build --packages-select pingpong_controller
source install/setup.bash

ros2 run pingpong_controller pingpong_node --ros-args \
  -p rl_policy_backend:=mjx \
  -p rl_model_path:=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/mjx_curriculum_last.pkl \
  -p robot_xml_path:=/home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  -p control_rate_hz:=200.0 \
  -p rl_policy_dt:=0.005 \
  -p realsense_fps:=60 \
  -p rl_ball_obs_age_clip:=0.5 \
  -p rl_action_gain:=0.0 \
  -p rl_action_scale_mult:=1.0 \
  -p rl_require_fk:=true \
  -p enable_init_pose:=true \
  -p init_pose_duration_s:=3.0 \
  -p record_rl_trace:=true
```

确认关节顺序、单位、坐标系、init pose、急停、`/joint_states`、安全 limiter 和 trace
均正常后，再在有人值守、无球条件下分级提高 action gain。首次接球前应核对：

1. `RightArm-0..6` 顺序，q/dq 为 rad/rad/s；
2. 球位置/速度为 `base_link` 下 m/m/s；
3. control timer 实测接近 200 Hz、相机新帧约 60 Hz；
4. 没有额外 software delay、actuator filter、inverse MPC 或 servo planner；
5. raw target 与 safe target 无持续饱和，实测 q/dq 无振荡或温升异常。

## 7. 提交拆分（不包含 test 代码）

工作区含大量其他实验，禁止 `git add .`。本次按要求只提交训练/验证所需共享代码、
最终模型、部署说明和关键结果，不提交任何 `test/` 文件，也不在本组命令中提交
`rl_2real`/ROS runtime 修改。训练共享源文件同时包含这一实验周期内相邻的 qvel/GPU1
profile；执行 7.1 前必须查看 cached diff。

注意：仅提交模型不能让旧版实物控制器自动理解 `action_command_mode=velocity`。当前
工作区中的 qvel runtime 适配仍是部署该 checkpoint 的必要条件，只是按本次范围不纳入
下面两个 commit；后续若需要归档，应单独审核、单独提交。

### 7.1 训练 companion stack

```bash
cd /home/yangzhe/Project/pingpong_controller
git branch --show-current

# 若此前执行过旧版指令，先把所有 test 文件从暂存区移出；不改工作区内容。
git restore --staged -- test

git add -- \
  pingpong_controller/tools/rl_sim/mjx_juggle_env.py \
  pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py \
  pingpong_controller/tools/rl_sim/train_juggle_mjx_ppo.py \
  pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py \
  pingpong_controller/tools/rl_sim/run_with_host_memory_guard.sh \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/launch_formal_gpu0_v14_online.sh \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/launch_video_gpu0_v14_final.sh \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/PLATEAU_REWORK_REPORT.md

git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git diff --cached --name-only -- test
git commit -m "sim: version GPU0 qvel v14 heavy-ball training stack"
```

### 7.2 模型与关键验证结果

```bash
cd /home/yangzhe/Project/pingpong_controller

git add -- \
  pingpong_controller/tools/rl_2real/GPU0_QVEL_V14_HEAVY_BALL_REAL_ROBOT_DEPLOYMENT.md \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/mjx_curriculum_last.pkl \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/final_promotion/heavy_3p7g_damp0p90.csv \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/final_video/README.md \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/final_video/gpu0_v14_final_heavy_3p7g_60hz.mp4 \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/final_video/action_plot.png \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/final_video/validation.csv

sha256sum \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/mjx_curriculum_last.pkl \
  pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/final_video/gpu0_v14_final_heavy_3p7g_60hz.mp4

git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git diff --cached --name-only -- test
git commit -m "model: release GPU0 qvel v14 heavy-ball checkpoint"
git show --stat --oneline HEAD
```

两个 commit 完成并核对 staged 文件后再执行：

```bash
git push origin main
```

明确排除：全部 `test/`、本次范围外的 `rl_2real`/ROS runtime、`models/moz1_pd.xml`
本地实验修改、标定 JSON、W&B cache、archive/best/interrupted checkpoint、大体积
action/obs trace、GPU1 continuation、`__pycache__` 和其他未选输出。

## 8. 当前验证状态

- `best`/`last` 同种子 64-episode 固定重球配对已完成，发布选择为 `last`。
- checkpoint SHA-256、最终视频 SHA-256 已复核。
- checkpoint 两步 NumPy controller + MuJoCo FK smoke test 已通过，动作模式和 67D
  观测契约与 checkpoint 一致。
- qvel 动作 helper、旧 acceleration 兼容、V12/V13/V14 课程和重球 validator 共
  12 个本地针对性测试通过；测试代码不属于本次提交范围。只有第三方 NetworkX 时间
  API deprecation warning。
- 部署/训练相关 Python `py_compile`、两个 launcher 的 `bash -n` 和目标文件
  `git diff --check` 均通过。
- 未声称整个仓库 full pytest 通过；工作区包含大量其他未提交实验，完整回归应在两个
  精确 commit 暂存并复核后另行运行。
