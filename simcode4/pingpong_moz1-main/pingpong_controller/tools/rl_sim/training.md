# Juggle Random Training Curriculum

Right-arm-only acceleration-command action policy. Obs dim = 50 (includes ball_obs_age).

## Stage 1a: Fixed Ball Hit Discovery

### 阶段目的

从 scratch 学会击中固定球路。dropout 关闭，ball_obs_age 维度存在但为 0。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1a \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.5 --ball-launch-height 0.30 \
  --ball-obs-rate-hz 100 --ball-obs-pos-noise-std 0.0 --ball-obs-vel-noise-std 0.0 \
  --ball-obs-noise-warmup-ratio 0.50 --ball-obs-noise-ramp-ratio 0.50 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.34 --posture-weight 0.02 --base-pose-weight 0.00 \
  --ball-base-x-penalty-weight 0.00 --ball-base-x-soft-limit 0.20 \
  --ball-base-vxy-penalty-weight 0.00 --torque-penalty-weight 0.00005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.00 --arm-acc-limit-penalty-weight 0.002 \
  --arm-limiter-penalty-weight 0.0 \
  --ball-spawn-cube-size 0.0 --ball-spawn-xy-jitter 0.0 --ball-spawn-z-jitter 0.0 \
  --ball-init-vxy-max 0.0 --ball-init-vz -0.28
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1a \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.5 --ball-launch-height 0.30 \
  --ball-obs-rate-hz 100 --ball-obs-pos-noise-std 0.0 --ball-obs-vel-noise-std 0.0 \
  --ball-obs-noise-warmup-ratio 0.50 --ball-obs-noise-ramp-ratio 0.50 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.34 --posture-weight 0.02 --base-pose-weight 0.00 \
  --ball-base-x-penalty-weight 0.00 --ball-base-x-soft-limit 0.20 \
  --ball-base-vxy-penalty-weight 0.00 --torque-penalty-weight 0.00005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.00 --arm-acc-limit-penalty-weight 0.002 \
  --arm-limiter-penalty-weight 0.0 \
  --ball-spawn-cube-size 0.0 --ball-spawn-xy-jitter 0.0 --ball-spawn-z-jitter 0.0 \
  --ball-init-vxy-max 0.0 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1a/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1a/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 1.5 \
  --ball-obs-rate-hz 100 --ball-obs-pos-noise-std 0.0 --ball-obs-vel-noise-std 0.0 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.34 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1a/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1a/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- ep_hits_max > 0, ep_hits_mean 开始非零
- arm_acc_ratio_max < 2
- 视频中能看到球被击起

---

## Stage 1b: Small Ball Init Randomization

### 阶段目的

引入小球初始扰动，避免策略只记死固定球路。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.4 --ball-launch-height 0.30 \
  --ball-obs-rate-hz 100 --ball-obs-pos-noise-std 0.0 --ball-obs-vel-noise-std 0.0 \
  --ball-obs-noise-warmup-ratio 0.50 --ball-obs-noise-ramp-ratio 0.50 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.36 --posture-weight 0.05 --base-pose-weight 0.00 \
  --ball-base-x-penalty-weight 0.00 --ball-base-x-soft-limit 0.20 \
  --ball-base-vxy-penalty-weight 0.00 --torque-penalty-weight 0.00005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.00 --arm-acc-limit-penalty-weight 0.005 \
  --arm-limiter-penalty-weight 0.0 \
  --ball-spawn-cube-size 0.02 --ball-spawn-xy-jitter 0.005 --ball-spawn-z-jitter 0.005 \
  --ball-init-vxy-max 0.003 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1a/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.4 --ball-launch-height 0.30 \
  --ball-obs-rate-hz 100 --ball-obs-pos-noise-std 0.0 --ball-obs-vel-noise-std 0.0 \
  --ball-obs-noise-warmup-ratio 0.50 --ball-obs-noise-ramp-ratio 0.50 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.36 --posture-weight 0.05 --base-pose-weight 0.00 \
  --ball-base-x-penalty-weight 0.00 --ball-base-x-soft-limit 0.20 \
  --ball-base-vxy-penalty-weight 0.00 --torque-penalty-weight 0.00005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.00 --arm-acc-limit-penalty-weight 0.005 \
  --arm-limiter-penalty-weight 0.0 \
  --ball-spawn-cube-size 0.02 --ball-spawn-xy-jitter 0.005 --ball-spawn-z-jitter 0.005 \
  --ball-init-vxy-max 0.003 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1b/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1b/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 1.4 \
  --ball-obs-rate-hz 100 --ball-obs-pos-noise-std 0.0 --ball-obs-vel-noise-std 0.0 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.36 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1b/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1b/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- ep_hits_mean 不掉回 0
- ep_hits_max 继续上升
- arm_vel_ratio_max / arm_acc_ratio_max 不爆

---

## Stage 1c: Center-Aware Ball Observation Noise Curriculum

### 阶段目的

加入 ball obs noise + 轻量 base-x 回中。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1c \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.25 --ball-launch-height 0.30 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.0015 --ball-obs-vel-noise-std 0.015 \
  --ball-obs-noise-warmup-ratio 0.20 --ball-obs-noise-ramp-ratio 0.30 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.38 --posture-weight 0.10 --base-pose-weight 0.00 \
  --ball-base-x-penalty-weight 0.30 --ball-base-x-soft-limit 0.20 \
  --ball-base-vxy-penalty-weight 0.06 --torque-penalty-weight 0.00008 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.03 --arm-acc-limit-penalty-weight 0.03 \
  --arm-limiter-penalty-weight 0.01 \
  --action-penalty-weight 0.0010 --action-delta-penalty-weight 0.0004 \
  --ball-spawn-cube-size 0.02 --ball-spawn-xy-jitter 0.005 --ball-spawn-z-jitter 0.005 \
  --ball-init-vxy-max 0.003 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1b/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1c \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.25 --ball-launch-height 0.30 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.0015 --ball-obs-vel-noise-std 0.015 \
  --ball-obs-noise-warmup-ratio 0.20 --ball-obs-noise-ramp-ratio 0.30 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.38 --posture-weight 0.10 --base-pose-weight 0.00 \
  --ball-base-x-penalty-weight 0.30 --ball-base-x-soft-limit 0.20 \
  --ball-base-vxy-penalty-weight 0.06 --torque-penalty-weight 0.00008 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.03 --arm-acc-limit-penalty-weight 0.03 \
  --arm-limiter-penalty-weight 0.01 \
  --action-penalty-weight 0.0010 --action-delta-penalty-weight 0.0004 \
  --ball-spawn-cube-size 0.02 --ball-spawn-xy-jitter 0.005 --ball-spawn-z-jitter 0.005 \
  --ball-init-vxy-max 0.003 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1c/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1c/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 1.25 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.0015 --ball-obs-vel-noise-std 0.015 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.38 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1c/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1c/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- ball_base_x_abs_mean 开始下降
- ep_hits_mean 保持非零
- arm_limiter_clip_frac_mean 不接近 1

---

## Stage 1d: Center-Aware Active Hit Transition

### 阶段目的

加强主动击球和回中。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1d \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.25 --ball-launch-height 0.31 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.002 --ball-obs-vel-noise-std 0.02 \
  --ball-obs-noise-warmup-ratio 0.15 --ball-obs-noise-ramp-ratio 0.25 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.40 --posture-weight 0.12 --base-pose-weight 0.03 \
  --ball-base-x-penalty-weight 0.70 --ball-base-x-soft-limit 0.15 \
  --ball-base-vxy-penalty-weight 0.08 --torque-penalty-weight 0.00003 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.02 --arm-acc-limit-penalty-weight 0.03 \
  --arm-limiter-penalty-weight 0.01 \
  --action-penalty-weight 0.0010 --action-delta-penalty-weight 0.0004 \
  --ball-spawn-cube-size 0.05 --ball-spawn-xy-jitter 0.012 --ball-spawn-z-jitter 0.015 \
  --ball-init-vxy-max 0.006 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1c/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1d \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.25 --ball-launch-height 0.31 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.002 --ball-obs-vel-noise-std 0.02 \
  --ball-obs-noise-warmup-ratio 0.15 --ball-obs-noise-ramp-ratio 0.25 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.40 --posture-weight 0.12 --base-pose-weight 0.03 \
  --ball-base-x-penalty-weight 0.70 --ball-base-x-soft-limit 0.15 \
  --ball-base-vxy-penalty-weight 0.08 --torque-penalty-weight 0.00003 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.02 --arm-acc-limit-penalty-weight 0.03 \
  --arm-limiter-penalty-weight 0.01 \
  --action-penalty-weight 0.0010 --action-delta-penalty-weight 0.0004 \
  --ball-spawn-cube-size 0.05 --ball-spawn-xy-jitter 0.012 --ball-spawn-z-jitter 0.015 \
  --ball-init-vxy-max 0.006 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1d/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1d/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 1.25 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.002 --ball-obs-vel-noise-std 0.02 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.40 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1d/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1d/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 稳定非零
- ball_base_x_abs_mean 不变大
- 击球后球反弹高度接近 target-height

---

## Stage 1e: Center-Aware Hit Consolidation

### 阶段目的

巩固稳定击球和回中。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1e \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.35 --base-pose-weight 0.08 \
  --ball-base-x-penalty-weight 1.50 --ball-base-x-soft-limit 0.10 \
  --ball-base-vxy-penalty-weight 0.12 --torque-penalty-weight 0.0003 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.10 --arm-acc-limit-penalty-weight 0.12 \
  --arm-limiter-penalty-weight 0.04 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0010 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1d/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1e \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.35 --base-pose-weight 0.08 \
  --ball-base-x-penalty-weight 1.50 --ball-base-x-soft-limit 0.10 \
  --ball-base-vxy-penalty-weight 0.12 --torque-penalty-weight 0.0003 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.10 --arm-acc-limit-penalty-weight 0.12 \
  --arm-limiter-penalty-weight 0.04 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0010 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1e/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1e/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 0.95 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1e/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1e/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 稳定 5+
- ball_base_x_abs_mean 明显低于偏侧
- arm_acc_ratio_max < 1.2

---

## Stage 1f: Hit Cadence Consolidation

### 阶段目的

加入 hit cadence、hit_min_count_interval、hit_reward_cap。后续阶段继续保留这些参数。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1f \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.35 --base-pose-weight 0.08 \
  --ball-base-x-penalty-weight 1.20 --ball-base-x-soft-limit 0.12 \
  --ball-base-vxy-penalty-weight 0.12 --torque-penalty-weight 0.0003 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.10 --arm-acc-limit-penalty-weight 0.12 \
  --arm-limiter-penalty-weight 0.04 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0010 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --hit-cadence-reward-weight 0.50 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.18 \
  --hit-min-interval-penalty-weight 1.00 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1e/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1f \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.35 --base-pose-weight 0.08 \
  --ball-base-x-penalty-weight 1.20 --ball-base-x-soft-limit 0.12 \
  --ball-base-vxy-penalty-weight 0.12 --torque-penalty-weight 0.0003 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.10 --arm-acc-limit-penalty-weight 0.12 \
  --arm-limiter-penalty-weight 0.04 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0010 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --hit-cadence-reward-weight 0.50 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.18 \
  --hit-min-interval-penalty-weight 1.00 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1f/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1f/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 0.95 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --hit-cadence-reward-weight 0.50 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.18 \
  --hit-min-interval-penalty-weight 1.00 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1f/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1f/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 稳定 5+
- counted_hit_interval_mean 接近 0.65s
- short_hit_interval_frac < 0.2
- ignored_fast_hit_frac < 0.3

---

## Stage 2a: Gentle Centering Transition

### 阶段目的

温和把球路往身体中间拉。保留 cadence/cap。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2a \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.60 --base-pose-weight 0.10 \
  --ball-base-x-penalty-weight 2.5 --ball-base-x-soft-limit 0.09 \
  --ball-base-vxy-penalty-weight 0.40 --torque-penalty-weight 0.0003 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.05 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.03 \
  --action-penalty-weight 0.0015 --action-delta-penalty-weight 0.0010 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1f/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 2000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2a \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.60 --base-pose-weight 0.10 \
  --ball-base-x-penalty-weight 2.5 --ball-base-x-soft-limit 0.09 \
  --ball-base-vxy-penalty-weight 0.40 --torque-penalty-weight 0.0003 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.05 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.03 \
  --action-penalty-weight 0.0015 --action-delta-penalty-weight 0.0010 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2a/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2a/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 0.95 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2a/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2a/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- ball_base_x_abs_mean 继续下降
- eval/mean_hits 不长期掉到 0~2

---

## Stage 2b: Centered Hit Consolidation

### 阶段目的

进一步压紧球路中心和动作可执行性。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.90 --base-pose-weight 0.20 \
  --ball-base-x-penalty-weight 4.0 --ball-base-x-soft-limit 0.07 \
  --ball-base-vxy-penalty-weight 0.80 --torque-penalty-weight 0.0004 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.05 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.04 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2a/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.90 --base-pose-weight 0.20 \
  --ball-base-x-penalty-weight 4.0 --ball-base-x-soft-limit 0.07 \
  --ball-base-vxy-penalty-weight 0.80 --torque-penalty-weight 0.0004 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.05 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.04 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2b/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2b/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 0.95 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2b/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2b/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- ball_base_x_abs_mean 进一步下降
- arm_acc_ratio_max / arm_vel_ratio_max < 1

---

## Stage 2c: Base-X Recenter With Mild Posture

### 阶段目的

强化 base-x 回中，温和姿态。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2c \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 1.00 --base-pose-weight 0.25 \
  --ball-base-x-penalty-weight 6.0 --ball-base-x-soft-limit 0.05 \
  --ball-base-vxy-penalty-weight 1.0 --torque-penalty-weight 0.0004 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.05 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.04 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.40 --ball-vxy-penalty-weight 0.10 \
  --racket-chest-xy-penalty-weight 0.40 --racket-chest-z-penalty-weight 0.35 \
  --racket-xy-gauss-reward-weight 0.20 --racket-xy-gauss-penalty-weight 0.20 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2b/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 2000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2c \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 1.00 --base-pose-weight 0.25 \
  --ball-base-x-penalty-weight 6.0 --ball-base-x-soft-limit 0.05 \
  --ball-base-vxy-penalty-weight 1.0 --torque-penalty-weight 0.0004 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.05 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.04 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.40 --ball-vxy-penalty-weight 0.10 \
  --racket-chest-xy-penalty-weight 0.40 --racket-chest-z-penalty-weight 0.35 \
  --racket-xy-gauss-reward-weight 0.20 --racket-xy-gauss-penalty-weight 0.20 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2c/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2c/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 0.95 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2c/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2c/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- ball_base_x_abs_mean 往 0.40~0.45 靠近
- eval/mean_hits >= 4

---

## Stage 3a: Smooth Hardware-Limited Action Consolidation

### 阶段目的

此阶段不再追求更多击球数。当前策略已经能稳定点球，Stage 3a 的目标是保持足够 hit 的同时，降低 ball_base_x 偏移、降低过快击球比例、保持硬件速度/加速度限制、让动作更平滑。

**关键变化：显式降低 hit reward base/combo**，不再鼓励追求更多击球数；只要求保持足够 hit（>= 5），同时优先优化回中、节奏、动作平滑和硬件限制。成功标准是稳定可执行，而不是 eval/mean_hits 继续增长。不开 DR。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3a \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.70 --base-pose-weight 0.10 \
  --ball-base-x-penalty-weight 8.0 --ball-base-x-soft-limit 0.05 \
  --ball-base-vxy-penalty-weight 1.50 --torque-penalty-weight 0.0005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0020 --action-delta-penalty-weight 0.0014 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.20 \
  --racket-chest-z-penalty-weight 0.15 \
  --hit-cadence-reward-weight 0.15 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 1.00 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --hit-reward-base 1.2 --hit-reward-combo 0.25 --center-flat-hit-reward-weight 1.2 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage2c/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 300000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3a \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.70 --base-pose-weight 0.10 \
  --ball-base-x-penalty-weight 8.0 --ball-base-x-soft-limit 0.05 \
  --ball-base-vxy-penalty-weight 1.50 --torque-penalty-weight 0.0005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0020 --action-delta-penalty-weight 0.0014 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.20 \
  --racket-chest-z-penalty-weight 0.15 \
  --hit-cadence-reward-weight 0.15 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 1.00 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --hit-reward-base 1.2 --hit-reward-combo 0.25 --center-flat-hit-reward-weight 1.2 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3a/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3a/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 0.90 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --ball-base-x-penalty-weight 8.0 --ball-base-x-soft-limit 0.05 \
  --ball-base-vxy-penalty-weight 1.50 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --hit-cadence-reward-weight 0.15 --hit-min-interval-penalty-weight 1.00 \
  --hit-min-interval 0.40 --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --hit-reward-base 1.2 --hit-reward-combo 0.25 --center-flat-hit-reward-weight 1.2 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3a/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3a/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 保持 >= 5 即可，不追求继续增长
- eval/ball_base_x_abs_mean < 0.20
- rollout/ball_base_x_abs_mean 有下降趋势，目标 < 0.20
- eval/ball_base_x_abs_max 尽量 < 0.40
- eval/short_hit_interval_frac < 0.12
- rollout/short_hit_interval_frac 继续下降
- eval/arm_acc_ratio_max < 1.5~2.0
- arm_cmd_acc_clip_frac_mean = 0
- action_delta_norm_mean 不上升

---

## Stage 3b: Light Camera Constraint

### 阶段目的

加轻量 D455 pixel 可见性约束。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 200000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.975 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 4.0 --camera-depth-penalty-weight 0.5 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --hit-reward-base 1.2 --hit-reward-combo 0.25 --center-flat-hit-reward-weight 1.2 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3a/best/best_model.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 2000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.95 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.25 \
  --camera-visibility-penalty-weight 1.0 --camera-depth-penalty-weight 0.5 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --hit-reward-base 1.2 --hit-reward-combo 0.25 --center-flat-hit-reward-weight 1.2 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3a/best/best_model.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 200000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 0.975 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --no-domain-randomization \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 4.0 --camera-depth-penalty-weight 0.5 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.30 --hit-cadence-target-interval 0.65 --hit-cadence-sigma 0.20 \
  --hit-min-interval-penalty-weight 0.80 --hit-min-interval 0.40 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --hit-reward-base 0.5 --hit-reward-combo 0.10 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3b/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3b/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu --no-domain-randomization \
  --action-acc-scale 0.975 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.42 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.25 \
  --camera-visibility-penalty-weight 1.0 --camera-depth-penalty-weight 0.5 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-min-count-interval 0.32 --fast-hit-penalty-weight 0.30 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.65 \
  --hit-reward-base 1.2 --hit-reward-combo 0.25 --center-flat-hit-reward-weight 1.2 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3b/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3b/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- camera_visible_mean > 0.8
- eval/mean_hits 稳定

---

## Stage 4a: Ball-Only Light DR

### 阶段目的

只加球本体轻随机化。

**备注：** 实际进入 Stage 4a 前，应确认 Stage 3b 最终 checkpoint 是否优于 last；如不是，请手动替换为对应 best/checkpoint。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4a \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset ball \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage3b_cadence050/best/best_model.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 8000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4a \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset ball \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4a/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4a/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu \
  --dr-preset ball \
  --action-acc-scale 1.0 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 \
  --posture-weight 0.80 --base-pose-weight 0.15 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4a/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4a/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 不明显下降
- dr_action_scale_mult_mean 接近 1.0

---

## Stage 4b: Contact DR

### 阶段目的

加球-拍接触随机化。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 200000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset ball_contact \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4a/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 8000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4b \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset ball_contact \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4b/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4b/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu \
  --dr-preset ball_contact \
  --action-acc-scale 1.0 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 \
  --posture-weight 0.80 --base-pose-weight 0.15 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4b/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4b/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 稳定
- 不频繁飞球

---

## Stage 4c: Lite Actuator DR

### 阶段目的

加轻度执行侧随机化。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 200000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4c \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset ball_contact_actuator \
  --dr-action-scale-mult-min 0.93 --dr-action-scale-mult-max 1.07 \
  --dr-damping-mult-min 0.85 --dr-damping-mult-max 1.15 \
  --dr-armature-mult-min 0.90 --dr-armature-mult-max 1.10 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4b/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4c \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset ball_contact_actuator \
  --dr-action-scale-mult-min 0.93 --dr-action-scale-mult-max 1.07 \
  --dr-damping-mult-min 0.85 --dr-damping-mult-max 1.15 \
  --dr-armature-mult-min 0.90 --dr-armature-mult-max 1.10 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4c/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4c/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu \
  --dr-preset ball_contact_actuator \
  --dr-action-scale-mult-min 0.93 --dr-action-scale-mult-max 1.07 \
  --dr-damping-mult-min 0.85 --dr-damping-mult-max 1.15 \
  --dr-armature-mult-min 0.90 --dr-armature-mult-max 1.10 \
  --action-acc-scale 1.0 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 \
  --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4c/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4c/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- 姿态仍合理，动作保持平滑
- eval/mean_hits 稳定

---

## Stage 4d: Latency DR

### 阶段目的

加 obs/action latency，sim2real 关键 gap。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4d \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset full \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4c/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 500000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4d \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 5.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset full \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 1.5 --hit-reward-combo 0.2 --center-flat-hit-reward-weight 1.5 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4d/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4d/ppo_juggle_dr_last.zip \
  --episodes 20 --deterministic --device cpu \
  --dr-preset full \
  --action-acc-scale 1.0 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 \
  --posture-weight 0.80 --base-pose-weight 0.15 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4d/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4d/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 在完整 DR + latency 下保持可用
- dr_obs_latency_steps_mean / dr_action_latency_steps_mean 非零

---

## Stage 4e: Racket Mount DR

### 阶段目的

适应球拍安装位置/角度/半径误差。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 3000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4e \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4d/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 3000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4e \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4e/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4e/ppo_juggle_dr_last.zip \
  --episodes 5 --deterministic --device cpu \
  --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --action-acc-scale 1.0 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.0 --ball-obs-dropout-max-steps 1 \
  --ball-obs-dropout-burst-prob 0.0 --ball-obs-dropout-burst-max-steps 1 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 \
  --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4e/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4e/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 稳定
- dr_racket_pos_offset_norm_mean 非零

---

## Stage 4f: Final DR + Camera + Ball Obs Dropout

### 阶段目的

最终 sim2real 候选。打开 ball obs dropout + camera pixel 约束。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 4000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4f \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.02 --ball-obs-dropout-max-steps 2 \
  --ball-obs-dropout-burst-prob 0.005 --ball-obs-dropout-burst-max-steps 5 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4e/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 4000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4f \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.02 --ball-obs-dropout-max-steps 2 \
  --ball-obs-dropout-burst-prob 0.005 --ball-obs-dropout-burst-max-steps 5 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4f/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4f/ppo_juggle_dr_last.zip \
  --episodes 20 --deterministic --device cpu \
  --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --action-acc-scale 1.0 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.02 --ball-obs-dropout-max-steps 2 \
  --ball-obs-dropout-burst-prob 0.005 --ball-obs-dropout-burst-max-steps 5 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 \
  --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 1 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 10.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 5.0 --camera-top-margin-penalty-weight 20.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.3 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.2 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.3 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4f/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4f/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_hits 在完整 DR + dropout 下稳定
- ball_obs_age_mean 非零但较小
- ball_obs_age_max < 0.10s
- ball_obs_dropout_frac 接近设定概率
- arm_limiter_clip_frac_mean < 0.2
- camera_visible_mean > 0.8
- 最终 sim2real 部署候选

---

## Stage 4g: Strong Contact DR

### 阶段目的

增强 ball/racket contact randomization，扩大 friction 和 solref 范围，作为 4f 后的额外 robustness 阶段。

### 从上一阶段开始训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4g \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.02 --ball-obs-dropout-max-steps 2 \
  --ball-obs-dropout-burst-prob 0.005 --ball-obs-dropout-burst-max-steps 5 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --dr-ball-friction-min 0.08 --dr-ball-friction-max 0.45 \
  --dr-racket-friction-min 0.18 --dr-racket-friction-max 0.75 \
  --dr-ball-solref-time-min 0.0015 --dr-ball-solref-time-max 0.010 \
  --dr-ball-solref-damping-min 0.55 --dr-ball-solref-damping-max 1.10 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4f/ppo_juggle_dr_last.zip
```

### 本阶段继续训练

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --n-envs 8 --device cpu --total-steps 1000000 \
  --save-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4g \
  --eval-freq 5000 --checkpoint-freq 12500 --frame-skip 5 \
  --action-acc-scale 1.0 --ball-launch-height 0.32 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-noise-warmup-ratio 0.10 --ball-obs-noise-ramp-ratio 0.20 \
  --ball-obs-dropout-prob 0.02 --ball-obs-dropout-max-steps 2 \
  --ball-obs-dropout-burst-prob 0.005 --ball-obs-dropout-burst-max-steps 5 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 --torque-penalty-weight 0.0005 \
  --seed 1 --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --dr-ball-friction-min 0.08 --dr-ball-friction-max 0.45 \
  --dr-racket-friction-min 0.18 --dr-racket-friction-max 0.75 \
  --dr-ball-solref-time-min 0.0015 --dr-ball-solref-time-max 0.010 \
  --dr-ball-solref-damping-min 0.55 --dr-ball-solref-damping-max 1.10 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-spawn-cube-size 0.10 --ball-spawn-xy-jitter 0.025 --ball-spawn-z-jitter 0.035 \
  --ball-init-vxy-max 0.012 --ball-init-vz -0.28 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --resume-from /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4g/ppo_juggle_dr_last.zip
```

### 验证

```bash
python3 /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/validate_juggle_rl_random.py \
  --xml /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --checkpoint /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4g/ppo_juggle_dr_last.zip \
  --episodes 20 --deterministic --device cpu \
  --dr-preset full \
  --dr-racket-mount --dr-racket-pos-offset-mm 3.0 --dr-racket-rot-offset-deg 1.0 --dr-racket-radius-offset-mm 2.0 \
  --action-acc-scale 1.0 \
  --ball-obs-rate-hz 50 --ball-obs-pos-noise-std 0.003 --ball-obs-vel-noise-std 0.03 \
  --ball-obs-dropout-prob 0.02 --ball-obs-dropout-max-steps 2 \
  --ball-obs-dropout-burst-prob 0.005 --ball-obs-dropout-burst-max-steps 5 \
  --ball-obs-age-clip 0.20 \
  --target-height 0.28 \
  --posture-weight 0.80 --base-pose-weight 0.15 \
  --ball-base-x-penalty-weight 1.0 --ball-base-x-soft-limit 0.025 \
  --ball-base-vxy-penalty-weight 6.0 \
  --dr-ball-friction-min 0.08 --dr-ball-friction-max 0.45 \
  --dr-racket-friction-min 0.18 --dr-racket-friction-max 0.75 \
  --dr-ball-solref-time-min 0.0015 --dr-ball-solref-time-max 0.010 \
  --dr-ball-solref-damping-min 0.55 --dr-ball-solref-damping-max 1.10 \
  --arm-action-limiter \
  --arm-vel-limit-deg-s 210 210 240 240 300 300 300 \
  --arm-acc-limit-deg-s2 1300 1300 1800 3000 3000 3000 3000 \
  --arm-vel-limit-penalty-weight 0.06 --arm-acc-limit-penalty-weight 0.08 \
  --arm-limiter-penalty-weight 0.08 \
  --action-penalty-weight 0.0018 --action-delta-penalty-weight 0.0012 \
  --ball-anchor-xy-penalty-weight 0.60 --racket-chest-xy-penalty-weight 0.55 \
  --racket-chest-z-penalty-weight 0.35 \
  --camera-visibility-mode pixel --camera-center-weight 0.5 \
  --camera-visibility-penalty-weight 8.0 --camera-depth-penalty-weight 0.5 \
  --camera-visible-penalty-weight 3.0 --camera-top-margin-penalty-weight 12.0 \
  --camera-pixel-margin 80 --camera-min-depth 0.15 --camera-max-depth 2.50 \
  --hit-cadence-reward-weight 0.05 --hit-cadence-target-interval 0.32 --hit-cadence-sigma 0.10 \
  --hit-min-interval-penalty-weight 1.50 --hit-min-interval 0.24 \
  --hit-min-count-interval 0.22 --fast-hit-penalty-weight 0.80 \
  --hit-reward-cap-mode auto --hit-reward-cap-target-interval 0.32 \
  --hit-reward-base 0.5 --hit-reward-combo 0.02 --center-flat-hit-reward-weight 0.8 \
  --record-video --video-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4g/video \
  --save-trajectory-dir /home/jacky/pingpong_ws/src/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage4g/traj \
  --video-fps 30 --video-width 640 --video-height 480 --headless \
  --log-hit-events --print-every 20
```

### 预期达到的目的

- eval/mean_ep_length 在 stronger contact DR 下尽量接近 1200
- eval/camera_visible_mean 保持 > 0.85，最好 > 0.90
- eval/mean_hits 不明显崩
- eval/arm_acc_ratio_max / eval/arm_vel_ratio_max 不持续超 1
- dr_ball_friction / dr_racket_friction / dr_ball_solref_* 范围生效
