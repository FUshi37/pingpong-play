#!/usr/bin/env bash
set -euo pipefail

SIMCODE4_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMCODE4_TREE="$SIMCODE4_WORKSPACE/simcode4/pingpong_moz1-main"
SIMCODE4_TRAIN="$SIMCODE4_TREE/pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py"
SIMCODE4_XML="$SIMCODE4_TREE/pingpong_controller/tools/rl_sim/moz1_pd.xml"
SIMCODE4_DEFAULT_CHECKPOINT="$SIMCODE4_TREE/pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/mjx_curriculum_best.pkl"

SIMCODE4_RESUME_FROM="${SIMCODE4_RESUME_FROM:-$SIMCODE4_DEFAULT_CHECKPOINT}"
SIMCODE4_CONDA_ENV="${SIMCODE4_CONDA_ENV:-pingpong_mjx}"
SIMCODE4_GPU="${SIMCODE4_GPU:-0}"
SIMCODE4_RESET_OPTIMIZER="${SIMCODE4_RESET_OPTIMIZER:-true}"
SIMCODE4_WANDB="${SIMCODE4_WANDB:-false}"
SIMCODE4_OUTPUT_DIR="${SIMCODE4_OUTPUT_DIR:-$SIMCODE4_WORKSPACE/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/simcode4_deployable_v6_$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "$SIMCODE4_RESUME_FROM" ]; then
    echo "ERROR: resume checkpoint not found: $SIMCODE4_RESUME_FROM" >&2
    exit 2
fi
if [ ! -f "$SIMCODE4_XML" ]; then
    echo "ERROR: simcode4 XML not found: $SIMCODE4_XML" >&2
    exit 2
fi

mkdir -p "$SIMCODE4_OUTPUT_DIR"

SIMCODE4_EXTRA_ARGS=()
if [ "$SIMCODE4_RESET_OPTIMIZER" = "true" ]; then
    SIMCODE4_EXTRA_ARGS+=(--reset-optimizer-on-resume)
fi
if [ "$SIMCODE4_WANDB" = "true" ]; then
    SIMCODE4_EXTRA_ARGS+=(
        --wandb
        --wandb-mode online
        --wandb-project pingpong-mjx
        --wandb-name simcode4-deployable-v6
        --wandb-tags goal-d455 deployable-v6 upstream-servo inverse-mpc
    )
fi

echo "[simcode4-v6] resume=$SIMCODE4_RESUME_FROM"
echo "[simcode4-v6] output=$SIMCODE4_OUTPUT_DIR"
echo "[simcode4-v6] plant=inverse-MPC -> planner -> physical-model delay/FOPDT -> XML PD"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="$SIMCODE4_GPU" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
conda run --no-capture-output -n "$SIMCODE4_CONDA_ENV" \
python "$SIMCODE4_TRAIN" \
    --xml "$SIMCODE4_XML" \
    --curriculum-profile goal_d455_autolaunch_viewdense_constrained_mpc_launch17_obsres2mm_servo_deployable_v6 \
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
    --resume-from "$SIMCODE4_RESUME_FROM" \
    --resume-start-stage launch19_final_measured_obsres2mm_servo_consolidation \
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
    --save-dir "$SIMCODE4_OUTPUT_DIR" \
    "${SIMCODE4_EXTRA_ARGS[@]}"
