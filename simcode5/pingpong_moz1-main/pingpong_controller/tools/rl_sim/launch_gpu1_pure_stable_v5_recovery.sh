#!/usr/bin/env bash
set -euo pipefail
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim

RUN_DIR=/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802
mkdir -p "$RUN_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

exec ./run_with_host_memory_guard.sh "$RUN_DIR/stdout_stderr.log" \
  /home/yangzhe/miniconda3/envs/pingpong/bin/python -u train_juggle_mjx_curriculum.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --curriculum-profile goal_d455_sport_taskspace_obsres2mm_nocomp_direct_v1 \
  --curriculum-gate-preset legacy \
  --delay-ablation-preset sport_actuator_replay_dr \
  --actuator-compensation-mode none \
  --resume-from /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_retreat_v4_20260802/mjx_curriculum_last.pkl \
  --resume-start-stage 17 --seed 976 \
  --n-envs 640 --n-steps 128 --minibatch-size 16384 \
  --update-epochs 4 --learning-rate 2e-4 --clip-range 0.15 \
  --gamma 0.9995 --gae-lambda 0.99 --ent-coef 5e-4 \
  --convergence-window 16 --convergence-min-episodes 64 \
  --advance-mode converged --advance-validation-mode block \
  --advance-eval-n-envs 192 --advance-eval-steps 1200 \
  --advance-eval-min-episodes 96 --advance-eval-reset-bucket-mode cvar \
  --mid-training-start-stage 6 --mid-n-steps 256 --mid-learning-rate 2e-4 \
  --mid-update-epochs 4 --mid-ent-coef 1e-4 --mid-convergence-window 20 \
  --late-optimizer-start-stage 18 --late-n-steps 256 --late-learning-rate 5e-5 \
  --late-update-epochs 2 --late-ent-coef 0 --late-convergence-window 24 \
  --save-every-updates 5 --archive-every-updates 25 \
  --gpu-max-temp-c 82 --gpu-check-every-updates 1 \
  --save-dir "$RUN_DIR" --wandb --wandb-mode offline \
  --wandb-project pingpong-mjx --wandb-name goal-d455-gpu1-pure-stable-v5-recovery
