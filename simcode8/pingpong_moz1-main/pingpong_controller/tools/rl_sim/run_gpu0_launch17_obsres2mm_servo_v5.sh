#!/usr/bin/env bash
set -o pipefail

RUN_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/goal_d455_gpu0_launch17_obsres2mm_servo_v5_unlimited_20260729
mkdir -p "${RUN_DIR}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="${RUN_DIR}/wandb"
export WANDB_CACHE_DIR="${RUN_DIR}/wandb_cache"
export WANDB_CONFIG_DIR="${RUN_DIR}/wandb_config"

/home/yangzhe/miniconda3/envs/pingpong/bin/python \
  /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --curriculum-profile goal_d455_autolaunch_viewdense_constrained_mpc_launch17_obsres2mm_servo_v5 \
  --curriculum-gate-preset v7_strict \
  --delay-ablation-preset real_actuator_replay_fit \
  --actuator-compensation-mode inverse_mpc \
  --actuator-mpc-feedback-source actual \
  --actuator-mpc-beta 1.2 \
  --actuator-mpc-delay-scale 1.05 \
  --actuator-mpc-tau-scale 0.75 \
  --actuator-mpc-horizon-steps 6 \
  --actuator-mpc-tracking-weight 1.0 \
  --actuator-mpc-nominal-weight 0.25 \
  --actuator-mpc-delta-weight 0.05 \
  --actuator-mpc-max-delta-deg 30 \
  --arm-servo-target-tracking-planner \
  --arm-servo-target-velocity-scale 1.0 \
  --arm-servo-target-acceleration-scale 0.8 \
  --asymmetric-critic \
  --critic-command-history-steps 12 \
  --resume-from /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/goal_d455_gpu0_launch17_gatefix_v5_unlimited_20260728/mjx_curriculum_best.pkl \
  --reset-optimizer-on-resume \
  --resume-start-stage 20 \
  --seed 976 \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 2 \
  --learning-rate 5e-5 \
  --gamma 0.9995 \
  --gae-lambda 0.99 \
  --time-limit-bootstrap \
  --failure-focus-hit-threshold 0 \
  --failure-focus-weight 1.0 \
  --failure-focus-tail-steps 0 \
  --clip-range 0.2 \
  --target-kl 0.012 \
  --ent-coef 0 \
  --min-log-std -4.8 \
  --actor-anchor-kl-coef 0 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  --advance-mode converged \
  --convergence-window 24 \
  --convergence-min-episodes 64 \
  --min-stage-updates 30 \
  --stage-metric-warmup-updates 5 \
  --max-stage-updates 0 \
  --max-stages 0 \
  --advance-validation-mode block \
  --advance-eval-stochastic \
  --advance-eval-n-envs 256 \
  --advance-eval-steps 1200 \
  --advance-eval-min-episodes 128 \
  --advance-eval-retry-updates 20 \
  --advance-eval-hit-ratio 1.0 \
  --advance-eval-len-ratio 1.0 \
  --advance-eval-min-hits 0.75 \
  --advance-eval-min-len-frac 0.10 \
  --advance-eval-min-return -1000000 \
  --advance-eval-hit-rate-margin 0.05 \
  --advance-eval-hit-interval-margin 0.03 \
  --advance-eval-cond-hit-ratio 0.95 \
  --advance-eval-camera-margin 0.05 \
  --advance-eval-ball-view-margin 0.05 \
  --advance-eval-z-ideal-margin 0.05 \
  --advance-eval-reset-bucket-mode cvar \
  --advance-eval-reset-bucket-min-episodes 8 \
  --advance-eval-reset-bucket-cvar-frac 0.25 \
  --advance-eval-reset-bucket-rate-margin 0.05 \
  --advance-eval-reset-bucket-hit-margin 1.0 \
  --save-every-updates 5 \
  --archive-every-updates 25 \
  --save-dir "${RUN_DIR}" \
  --wandb \
  --wandb-mode online \
  --wandb-project pingpong-mjx \
  --wandb-name goal-d455-gpu0-launch17-obsres2mm-servo-v5-unlimited \
  --wandb-tags goal-d455 gpu0 launch17 measured-obsres2mm actual-feedback servo-planner inverse-mpc online \
  2>&1 | tee "${RUN_DIR}/stdout_stderr.log"
