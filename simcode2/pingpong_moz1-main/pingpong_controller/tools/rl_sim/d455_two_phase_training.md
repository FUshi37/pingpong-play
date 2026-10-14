# D455 two-phase juggling training

目标是先训练一个固定右臂 reset、稳定主轨迹的 D455 策略，再从这个 checkpoint 进入 recovery-state learning。这样把“学会正常颠球”和“学会从偏离状态恢复”拆开，避免从一开始就让 PPO 同时解决首拍搜索、连续颠球、missing 和大范围恢复。

注意：PPO 本身不是 replay buffer 方法；这里的问题更准确地说是 on-policy state visitation 后期变窄，失败/偏离状态被采样得越来越少。第二阶段用 recovery-state sampling 补这个分布缺口。

## 代码文件

提交到另一台机器时至少包含这些文件：

- `pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py`
  - 新增 `d455_stable_4g_v1` 和 `d455_recovery_v1`。
  - 两个 profile 都自动固定 67D actor、`real_actuator_replay_fit`、`inverse_mpc`、asymmetric critic history 12。
- `pingpong_controller/tools/rl_sim/mjx_juggle_env.py`
  - 使用 `anchor_drop` 和 `falling_contact` reset。
  - 包含 hit 后 apex/view center 与 next-contact anchor recoverability reward/metric。
- `pingpong_controller/tools/rl_sim/camera_calibration.py`
  - D455 848x480 undistorted intrinsics/extrinsics 常量。
- `pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py`
  - 用于训练后 deterministic/stochastic 验证。
- `pingpong_controller/tools/rl_sim/moz1_pd.xml`
  - 训练 XML。

## Phase A: stable 4g from scratch

课程 profile: `d455_stable_4g_v1`

关键设计：

- 右臂 reset 固定为新的实物角度 `5.736 -44.399 30.683 97.142 49.323 -12.269 14.214`。
- 2026-07-13 复核发现 D455 外参在当前 MJCF 中挂 `waist02` 会把该 reset 的球拍点投到图像顶部；挂 `waist03` 时 `right_ee_site` 投影 `v_frac≈0.65`，与实物球在球拍中心的视觉结果一致。因此 `camera_calibration.py` 中 `D455_848_UNDISTORTED_SIM_BASE_BODY="waist03"`。
- 新角度在当前 XML 中 `right_racket` 的 MuJoCo world 坐标约 `[-0.114,-0.467,1.052]m`，`right_ee_site` 约 `[-0.114,-0.467,1.059]m`；如果减去 XML base body 的 `0.100m` z 偏移，则与实物 `right_racket/right_ee_site` 的 z 基本对齐。仿真训练坐标系不改。
- 最新实物有效视野范围从第一阶段开始使用：`x=[-0.25,0.25]m`，`y=[-0.50,-0.20]m`，`z=[1.00,1.47]m`。stable 的 z ideal 为 `[1.02,1.30]m`，recovery 的 z ideal 为 `[1.00,1.32]m`，`ball_view_y_target=-0.35m`。2026-07-13 确认实测 x/y 已经是 XML base 系，实测 z 是 XML/world z 减去 `0.100m` base 高度；因此训练中的 `ball_view_z_bounds_m` 用实测 z 加 `0.100m` 后的 `[1.00,1.47]m`，但不整体上移 hit/ideal 高度目标。
- 球 reset 使用 `anchor_drop`，不使用 falling-contact recovery sampling。
- 从第一个 stage 开始使用最新 D455 pixel camera。
- 不开启 camera missing/dropout；只保留小观测噪声。
- D455 投影使用 OpenCV optical 坐标：`v = cy + fy * y_cam / z_cam`。在 `waist03` 挂载下，球拍中心约 `v_frac=0.65`，拍上方 0.20m 约 `v_frac=0.41`。
- 击球位置约束在 D455 中部到中下方：`hit_camera_target_v_frac=0.66`，lower band `0.50..0.82`。
- 目标锚点从第一阶段开始就在新实物 reset 球拍附近：第一阶段 actual anchor 约 `y=-0.469..-0.445,z=1.046..1.062`；final stable 约 `y=-0.395..-0.306,z=1.012..1.085`。
- 球初始高度保持在实际球拍上方约 `0.19..0.25m`，避免离拍面过近，同时不使用“先高位训练再压低”的课程。
- 最后两个 stage 是 `stage4g_d455_stable_dr` 和 `stage4h_d455_stable_polish`。

2026-07-13 的 reset 投影检查：

- `stage1a_d455_anchor_drop_first_hit`: actual racket 约 `x=-0.112,y=-0.464,z=1.052`；target anchor 约 `y=-0.469..-0.445,z=1.046..1.062`；reset ball `z` 约 `1.244..1.265`，ball-racket `z` 平均约 `0.202`，`xy` 平均约 `0.016`，visible `1.0`，`v_frac` 约 `0.386..0.436`。
- `stage4h_d455_stable_polish`: target anchor 约 `y=-0.392..-0.306,z=1.012..1.083`；reset ball `z` 约 `1.232..1.320`，ball-racket `z` 平均约 `0.222`，`xy` 平均约 `0.121`，visible `1.0`，`v_frac` 约 `0.360..0.637`。

GPU0 稳定版：

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed972_gpu0/wandb \
WANDB_CACHE_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed972_gpu0/wandb_cache \
WANDB_CONFIG_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed972_gpu0/wandb_config \
/home/yangzhe/miniconda3/envs/pingpong/bin/python train_juggle_mjx_curriculum.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --curriculum-profile d455_stable_4g_v1 \
  --curriculum-gate-preset legacy \
  --delay-ablation-preset real_actuator_replay_fit \
  --actuator-compensation-mode inverse_mpc \
  --actuator-mpc-beta 1.2 \
  --actuator-mpc-delay-scale 1.05 \
  --actuator-mpc-tau-scale 0.75 \
  --actuator-mpc-horizon-steps 6 \
  --actuator-mpc-tracking-weight 1.0 \
  --actuator-mpc-nominal-weight 0.25 \
  --actuator-mpc-delta-weight 0.05 \
  --actuator-mpc-max-delta-deg 30 \
  --asymmetric-critic \
  --critic-command-history-steps 12 \
  --seed 972 \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 3 \
  --learning-rate 2e-4 \
  --clip-range 0.15 \
  --convergence-window 24 \
  --min-stage-updates 60 \
  --max-stage-updates 2000 \
  --advance-mode converged \
  --advance-validation-mode block \
  --advance-eval-stochastic \
  --advance-eval-ball-view-margin 0.05 \
  --advance-eval-z-ideal-margin 0.05 \
  --advance-eval-hit-ratio 0.45 \
  --advance-eval-min-hits 2.5 \
  --advance-eval-hit-rate-margin 0.05 \
  --advance-eval-hit-interval-margin 0.05 \
  --advance-eval-cond-hit-ratio 0.75 \
  --advance-eval-reset-bucket-mode cvar \
  --advance-eval-reset-bucket-min-episodes 4 \
  --advance-eval-reset-bucket-cvar-frac 0.20 \
  --advance-eval-reset-bucket-rate-margin 0.08 \
  --advance-eval-reset-bucket-hit-margin 1.0 \
  --save-every-updates 10 \
  --archive-every-updates 50 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed972_gpu0 \
  --wandb \
  --wandb-mode offline \
  --wandb-project pingpong-mjx \
  --wandb-name d455-stable-4g-v1-realview-zplus100-seed972-gpu0
```

GPU1 吞吐版：

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed973_gpu1/wandb \
WANDB_CACHE_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed973_gpu1/wandb_cache \
WANDB_CONFIG_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed973_gpu1/wandb_config \
/home/yangzhe/miniconda3/envs/pingpong/bin/python train_juggle_mjx_curriculum.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --curriculum-profile d455_stable_4g_v1 \
  --curriculum-gate-preset legacy \
  --delay-ablation-preset real_actuator_replay_fit \
  --actuator-compensation-mode inverse_mpc \
  --actuator-mpc-beta 1.2 \
  --actuator-mpc-delay-scale 1.05 \
  --actuator-mpc-tau-scale 0.75 \
  --actuator-mpc-horizon-steps 6 \
  --actuator-mpc-tracking-weight 1.0 \
  --actuator-mpc-nominal-weight 0.25 \
  --actuator-mpc-delta-weight 0.05 \
  --actuator-mpc-max-delta-deg 30 \
  --asymmetric-critic \
  --critic-command-history-steps 12 \
  --seed 973 \
  --n-envs 768 \
  --n-steps 384 \
  --minibatch-size 18432 \
  --update-epochs 2 \
  --learning-rate 2e-4 \
  --clip-range 0.15 \
  --convergence-window 24 \
  --min-stage-updates 60 \
  --max-stage-updates 2000 \
  --advance-mode converged \
  --advance-validation-mode block \
  --advance-eval-stochastic \
  --advance-eval-ball-view-margin 0.05 \
  --advance-eval-z-ideal-margin 0.05 \
  --advance-eval-hit-ratio 0.45 \
  --advance-eval-min-hits 2.5 \
  --advance-eval-hit-rate-margin 0.05 \
  --advance-eval-hit-interval-margin 0.05 \
  --advance-eval-cond-hit-ratio 0.75 \
  --advance-eval-reset-bucket-mode cvar \
  --advance-eval-reset-bucket-min-episodes 4 \
  --advance-eval-reset-bucket-cvar-frac 0.20 \
  --advance-eval-reset-bucket-rate-margin 0.08 \
  --advance-eval-reset-bucket-hit-margin 1.0 \
  --save-every-updates 10 \
  --archive-every-updates 50 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed973_gpu1 \
  --wandb \
  --wandb-mode offline \
  --wandb-project pingpong-mjx \
  --wandb-name d455-stable-4g-v1-realview-zplus100-seed973-gpu1
```

进入 Phase B 前的最低接受标准：

- 已到 `stage4h_d455_stable_polish`，不要从早期 stage 直接进入 recovery。
- `mean_hits` 稳定在 13 左右，最好 13-15。
- `mean_len_frac >= 0.93`，对应 1200 step 中约 1116 step 以上；最好接近 1200。
- `hit12_rate >= 0.78`，`target_episode_truncation_rate` 过线。
- `hit_camera_lower_band_rate` 高，且 `mean_hit_vxy`/`mean_hit_next_contact_anchor_err` 没有明显恶化。
- deterministic 和 stochastic validate 都没有 0-hit bucket。

如果 stable 已经到 `stage4g_d455_stable_dr` 但 polish 没过，先从该 checkpoint 继续 polish：

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/yangzhe/miniconda3/envs/pingpong/bin/python train_juggle_mjx_curriculum.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --resume-from /path/to/09_stage4g_d455_stable_dr.pkl \
  --resume-start-stage stage4h_d455_stable_polish \
  --curriculum-profile d455_stable_4g_v1 \
  --curriculum-gate-preset legacy \
  --delay-ablation-preset real_actuator_replay_fit \
  --actuator-compensation-mode inverse_mpc \
  --actuator-mpc-beta 1.2 \
  --actuator-mpc-delay-scale 1.05 \
  --actuator-mpc-tau-scale 0.75 \
  --actuator-mpc-horizon-steps 6 \
  --actuator-mpc-tracking-weight 1.0 \
  --actuator-mpc-nominal-weight 0.25 \
  --actuator-mpc-delta-weight 0.05 \
  --actuator-mpc-max-delta-deg 30 \
  --asymmetric-critic \
  --critic-command-history-steps 12 \
  --seed 910 \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --clip-range 0.10 \
  --convergence-window 24 \
  --min-stage-updates 80 \
  --max-stage-updates 2500 \
  --advance-mode converged \
  --advance-validation-mode block \
  --advance-eval-stochastic \
  --save-every-updates 10 \
  --archive-every-updates 50 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_polish_resume_seed910_gpu0 \
  --wandb \
  --wandb-entity fushi37 \
  --wandb-project pingpong-mjx \
  --wandb-name d455-stable-4g-v1-polish-resume-seed910-gpu0
```

## Phase B: recovery learning

课程 profile: `d455_recovery_v1`

只从 Phase A 的 `stage4h_d455_stable_polish` 合格 checkpoint resume。不要从零训练这个 profile。

关键设计：

- 右臂 reset 仍固定。
- 球 reset 改成 `falling_contact`：从不同高度、位置、横向速度的 post-apex 下落状态开始。
- reward 明确鼓励击球后竖直、回到 D455 中下方、下一次落点接近球拍 anchor。
- missing、dropout、观测噪声和 DR 逐步增大。
- `recovery1` 仍保持 reset 可见，先学可见偏离恢复；`recovery2..5` 再逐步扩大范围并加入 missing。
- final stage 是 `recovery5_final_missing_polish`。

2026-07-13 的 recovery reset 投影检查：

- `recovery1_descent_small_no_missing`: actual racket 约 `y=-0.464,z=1.052`；target anchor 约 `y=-0.419..-0.350,z=1.022..1.080`；reset ball `z` 约 `1.205..1.334`，visible `1.0`，`v_frac` 约 `0.290..0.621`，reset missing `0.0`。
- `recovery5_final_missing_polish`: target anchor 约 `y=-0.438..-0.274,z=0.985..1.106`，reset ball `z` 约 `1.205..1.434`，visible `1.0`，`v_frac` 约 `0.072..0.883`；missing/dropout 由观测管线逐步引入。

先选择 stable checkpoint：

```bash
STABLE_CKPT=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed972_gpu0/mjx_curriculum_best.pkl
```

GPU0 recovery 稳定版：

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_recovery_v1_realview_zplus100_from_stable_seed982_gpu0/wandb \
WANDB_CACHE_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_recovery_v1_realview_zplus100_from_stable_seed982_gpu0/wandb_cache \
WANDB_CONFIG_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_recovery_v1_realview_zplus100_from_stable_seed982_gpu0/wandb_config \
/home/yangzhe/miniconda3/envs/pingpong/bin/python train_juggle_mjx_curriculum.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --resume-from "${STABLE_CKPT}" \
  --resume-start-stage recovery1_descent_small_no_missing \
  --curriculum-profile d455_recovery_v1 \
  --curriculum-gate-preset legacy \
  --delay-ablation-preset real_actuator_replay_fit \
  --actuator-compensation-mode inverse_mpc \
  --actuator-mpc-beta 1.2 \
  --actuator-mpc-delay-scale 1.05 \
  --actuator-mpc-tau-scale 0.75 \
  --actuator-mpc-horizon-steps 6 \
  --actuator-mpc-tracking-weight 1.0 \
  --actuator-mpc-nominal-weight 0.25 \
  --actuator-mpc-delta-weight 0.05 \
  --actuator-mpc-max-delta-deg 30 \
  --asymmetric-critic \
  --critic-command-history-steps 12 \
  --seed 982 \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 3 \
  --learning-rate 1e-4 \
  --clip-range 0.10 \
  --convergence-window 24 \
  --min-stage-updates 80 \
  --max-stage-updates 2500 \
  --advance-mode converged \
  --advance-validation-mode block \
  --advance-eval-stochastic \
  --advance-eval-ball-view-margin 0.05 \
  --advance-eval-z-ideal-margin 0.05 \
  --advance-eval-hit-ratio 0.45 \
  --advance-eval-min-hits 2.5 \
  --advance-eval-hit-rate-margin 0.05 \
  --advance-eval-hit-interval-margin 0.05 \
  --advance-eval-cond-hit-ratio 0.75 \
  --advance-eval-reset-bucket-mode cvar \
  --advance-eval-reset-bucket-min-episodes 4 \
  --advance-eval-reset-bucket-cvar-frac 0.20 \
  --advance-eval-reset-bucket-rate-margin 0.08 \
  --advance-eval-reset-bucket-hit-margin 1.0 \
  --save-every-updates 10 \
  --archive-every-updates 50 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_recovery_v1_realview_zplus100_from_stable_seed982_gpu0 \
  --wandb \
  --wandb-entity fushi37 \
  --wandb-project pingpong-mjx \
  --wandb-name d455-recovery-v1-realview-zplus100-from-stable-seed982-gpu0
```

GPU1 recovery 吞吐版：

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
WANDB_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_recovery_v1_realview_zplus100_from_stable_seed983_gpu1/wandb \
WANDB_CACHE_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_recovery_v1_realview_zplus100_from_stable_seed983_gpu1/wandb_cache \
WANDB_CONFIG_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_recovery_v1_realview_zplus100_from_stable_seed983_gpu1/wandb_config \
/home/yangzhe/miniconda3/envs/pingpong/bin/python train_juggle_mjx_curriculum.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --resume-from "${STABLE_CKPT}" \
  --resume-start-stage recovery1_descent_small_no_missing \
  --curriculum-profile d455_recovery_v1 \
  --curriculum-gate-preset legacy \
  --delay-ablation-preset real_actuator_replay_fit \
  --actuator-compensation-mode inverse_mpc \
  --actuator-mpc-beta 1.2 \
  --actuator-mpc-delay-scale 1.05 \
  --actuator-mpc-tau-scale 0.75 \
  --actuator-mpc-horizon-steps 6 \
  --actuator-mpc-tracking-weight 1.0 \
  --actuator-mpc-nominal-weight 0.25 \
  --actuator-mpc-delta-weight 0.05 \
  --actuator-mpc-max-delta-deg 30 \
  --asymmetric-critic \
  --critic-command-history-steps 12 \
  --seed 983 \
  --n-envs 768 \
  --n-steps 384 \
  --minibatch-size 18432 \
  --update-epochs 2 \
  --learning-rate 1.5e-4 \
  --clip-range 0.12 \
  --convergence-window 24 \
  --min-stage-updates 80 \
  --max-stage-updates 2500 \
  --advance-mode converged \
  --advance-validation-mode block \
  --advance-eval-stochastic \
  --advance-eval-ball-view-margin 0.05 \
  --advance-eval-z-ideal-margin 0.05 \
  --advance-eval-hit-ratio 0.45 \
  --advance-eval-min-hits 2.5 \
  --advance-eval-hit-rate-margin 0.05 \
  --advance-eval-hit-interval-margin 0.05 \
  --advance-eval-cond-hit-ratio 0.75 \
  --advance-eval-reset-bucket-mode cvar \
  --advance-eval-reset-bucket-min-episodes 4 \
  --advance-eval-reset-bucket-cvar-frac 0.20 \
  --advance-eval-reset-bucket-rate-margin 0.08 \
  --advance-eval-reset-bucket-hit-margin 1.0 \
  --save-every-updates 10 \
  --archive-every-updates 50 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_recovery_v1_realview_zplus100_from_stable_seed983_gpu1 \
  --wandb \
  --wandb-entity fushi37 \
  --wandb-project pingpong-mjx \
  --wandb-name d455-recovery-v1-realview-zplus100-from-stable-seed983-gpu1
```

## Validation

Stable 通过后做 deterministic validate：

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/yangzhe/miniconda3/envs/pingpong/bin/python validate_juggle_mjx_ppo.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed972_gpu0/mjx_curriculum_best.pkl \
  --episodes 128 \
  --n-envs 128 \
  --deterministic \
  --results-csv /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/d455_stable_4g_v1_realview_zplus100_seed972_gpu0/validate_det_128.csv
```

Recovery 通过后再做 deterministic 和 stochastic validate。stochastic 版本去掉 `--deterministic` 即可。

## Monitoring

每 15-30 分钟看一次：

- `curriculum_progress.csv`: `stage`, `update`, `mean_hits`, `mean_episode_length`, `hit12_rate`, `episode_truncation_rate`。
- D455 相关：`camera_visible`, `ball_view_in_bounds`, `ball_view_z_ideal`, `hit_camera_visible_rate`, `hit_camera_lower_band_rate`。
- recovery 相关：`mean_hit_vxy`, `mean_hit_next_contact_anchor_err`, `mean_hit_camera_v_frac`。
- missing 相关：`ball_obs_missing_refresh_rate`, `ball_obs_lost_rate`, `ball_obs_age`。

判断原则：

- 没到 `min_updates` 或曲线还在明显上升，不要修改课程。
- 某 stage 达到 `max-stage-updates` 仍不过 gate，再先看 validate bucket 和 done term，再决定是继续加 update 上限还是只调当前 stage。
- 如果 stable 阶段还没稳定 13-15 hits，不要启动 recovery。
- 如果 recovery 后稳定主轨迹退化，先回退到前一个 recovery checkpoint，用更低 learning rate 或更小 recovery range 继续。
