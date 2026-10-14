#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
PYTHON_BIN=${PYTHON_BIN:-/home/yangzhe/miniconda3/envs/pingpong/bin/python}
XML_PATH=${XML_PATH:-$RL_SIM/moz1_pd.xml}
RMP_EVIDENCE_DIR=$REPO_ROOT/pingpong_controller/outputs/rl_sim/rmp_measured_qvel_design_20260820
RMP_OUTPUT_REPLAY_REPORT=${RMP_OUTPUT_REPLAY_REPORT:-$RMP_EVIDENCE_DIR/replay_datatracer_training_local_v2.json}
QREF_SCREEN=$REPO_ROOT/pingpong_controller/outputs/rl_sim/qref_horizon_screen_20260821_v1/screen.json
DEFAULT_SOURCE_RUN_DIR=$REPO_ROOT/pingpong_controller/outputs/rl_sim
DEFAULT_SOURCE_RUN_DIR+=/measured_qvel_rmp_vertical_v36_gpu0_seed20260822_20260824_complete_dr_critic_resume_v31_stage23
SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-$DEFAULT_SOURCE_RUN_DIR}
DEFAULT_RESUME_CHECKPOINT=$SOURCE_RUN_DIR/mjx_curriculum_best.pkl
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-$DEFAULT_RESUME_CHECKPOINT}
EXPECTED_RESUME_SHA256=${EXPECTED_RESUME_SHA256:-eb6cf8c1c71c5438aa3ef0eb60a1965f7636bc1990b08a6020f27b2f10a8520d}
RESUME_STAGE=${RESUME_STAGE:-rmp37_nonexecution_recovery_small_execution}
NEW_BALL_MASS_MIN_KG=${NEW_BALL_MASS_MIN_KG:-0.0030}
NEW_BALL_MASS_MAX_KG=${NEW_BALL_MASS_MAX_KG:-0.0040}
MAX_STAGES=${MAX_STAGES:-28}
N_ENVS=${N_ENVS:-1024}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-16384}
LEARNING_RATE=${LEARNING_RATE:-1.0e-4}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-3}
CLIP_RANGE=${CLIP_RANGE:-0.15}
TARGET_KL=${TARGET_KL:-0.008}
CONVERGENCE_WINDOW=${CONVERGENCE_WINDOW:-24}
STAGE_METRIC_WARMUP_UPDATES=${STAGE_METRIC_WARMUP_UPDATES:-10}
ADVANCE_EVAL_N_ENVS=${ADVANCE_EVAL_N_ENVS:-128}
CLEAR_JAX_CACHES_BETWEEN_STAGES=${CLEAR_JAX_CACHES_BETWEEN_STAGES:-NO}
TRANSACTIONAL_MAIN_METRIC_GUARD=${TRANSACTIONAL_MAIN_METRIC_GUARD:-NO}
TRANSACTION_INTERVAL_UPDATES=${TRANSACTION_INTERVAL_UPDATES:-8}
TRANSACTION_EVAL_N_ENVS=${TRANSACTION_EVAL_N_ENVS:-128}
TRANSACTION_EVAL_SEED=${TRANSACTION_EVAL_SEED:-260909}
TRANSACTION_MIN_PRIMARY_GAP_IMPROVEMENT=${TRANSACTION_MIN_PRIMARY_GAP_IMPROVEMENT:-0.0}
TRANSACTION_PRIMARY_FULL_RATE_TOLERANCE=${TRANSACTION_PRIMARY_FULL_RATE_TOLERANCE:-0.0}
TRANSACTION_PRIMARY_VALUE_NONREGRESSION=${TRANSACTION_PRIMARY_VALUE_NONREGRESSION:-NO}
TRANSACTION_SECONDARY_GAP_TOLERANCE=${TRANSACTION_SECONDARY_GAP_TOLERANCE:-1.0e-12}
TRANSACTION_MAX_ARM_QVEL_LIMIT_EXCEED_FRACTION=${TRANSACTION_MAX_ARM_QVEL_LIMIT_EXCEED_FRACTION:-1.0}
TRANSACTION_MAX_ARM_QACC_LIMIT_EXCEED_FRACTION=${TRANSACTION_MAX_ARM_QACC_LIMIT_EXCEED_FRACTION:-1.0}
# Historical/default command contract: --max-stage-updates -1.
MAX_STAGE_UPDATES=${MAX_STAGE_UPDATES:--1}
LOCAL_OFFLINE_SMOKE=${LOCAL_OFFLINE_SMOKE:-NO}
LAUNCHER_AUDIT_PATH=${LAUNCHER_AUDIT_PATH:-$RL_SIM/run_gpu0_measured_qvel_rmp_vertical_v37_resume_v36.sh}

RUN_SEED=${RUN_SEED:-20260822}
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d)}
DEFAULT_RUN_ID=measured_qvel_rmp_vertical_v37_gpu0
DEFAULT_RUN_ID+=_seed${RUN_SEED}_${RUN_DATE}_nonexecution_first_resume_v36
RUN_ID=${RUN_ID:-$DEFAULT_RUN_ID}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/$RUN_ID}
WANDB_MODE=${WANDB_MODE:-online}
if [[ "$LOCAL_OFFLINE_SMOKE" == YES ]]; then
  WANDB_MODE=offline
fi
# The user explicitly authorized V37 to continue V36's W&B run. V36's final
# saved/history point is true/W&B step 2276458496/2955149312. The selected
# V36 best checkpoint is older (true step 2221146112), so offset 734003200
# maps V37's first rollout to W&B step 2955280384 without changing checkpoint
# or optimizer time.
WANDB_ID=${WANDB_ID:-9acnp70r}
WANDB_RESUME=${WANDB_RESUME:-must}
WANDB_STEP_OFFSET=${WANDB_STEP_OFFSET:-734003200}
WANDB_NAME=${WANDB_NAME:-measured_qvel_rmp_vertical_v15_gpu0_seed20260821_20260821_formal1}
EXPECTED_GPU_UUID=GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4
PROFILE=goal_d455_measured_qvel_rmp_vertical_v37
PROFILE=${PROFILE_OVERRIDE:-$PROFILE}
PROFILE_TAG=${PROFILE_TAG:-v37}
RESUME_SOURCE_TAG=${RESUME_SOURCE_TAG:-resume-v36-stage26}
SOURCE_SELECTION_AUDIT=${SOURCE_SELECTION_AUDIT:-Screened V36 Stage-26 best, true step 2221146112, update 303.}
SOURCE_METRIC_AUDIT=${SOURCE_METRIC_AUDIT:-Screen: hits=14.180, full=0.906, mean_len=1115.9/1200, view=0.967.}
SOURCE_STATE_AUDIT=${SOURCE_STATE_AUDIT:-State: unchanged 57-D actor, 368-D complete-DR critic and Adam moments.}
STAGE_20_AUDIT=${STAGE_20_AUDIT:-Stage 20: exact conv_len=1.0/full=1.0 polish; passing archive is the runnable model.}
POLISH_TAG=${POLISH_TAG:-exact-full-episode-polish}

OBS_DIM_MIGRATION_ARGS=("${OBS_DIM_MIGRATION_FLAG:---no-allow-obs-dim-migration}")
CURRICULUM_RESUME_ARGS=()
if [[ -n "${CURRICULUM_RESUME_FLAG:-}" ]]; then
  CURRICULUM_RESUME_ARGS=("$CURRICULUM_RESUME_FLAG")
fi
CLEAR_STAGE_CACHE_ARGS=(--no-clear-jax-caches-between-stages)
if [[ "$CLEAR_JAX_CACHES_BETWEEN_STAGES" == YES ]]; then
  CLEAR_STAGE_CACHE_ARGS=(--clear-jax-caches-between-stages)
elif [[ "$CLEAR_JAX_CACHES_BETWEEN_STAGES" != NO ]]; then
  echo "CLEAR_JAX_CACHES_BETWEEN_STAGES must be YES or NO" >&2
  exit 2
fi
TRANSACTION_ARGS=()
if [[ "$TRANSACTIONAL_MAIN_METRIC_GUARD" == YES ]]; then
  TRANSACTION_PRIMARY_VALUE_ARGS=(--no-transaction-primary-value-nonregression)
  if [[ "$TRANSACTION_PRIMARY_VALUE_NONREGRESSION" == YES ]]; then
    TRANSACTION_PRIMARY_VALUE_ARGS=(--transaction-primary-value-nonregression)
  elif [[ "$TRANSACTION_PRIMARY_VALUE_NONREGRESSION" != NO ]]; then
    echo "TRANSACTION_PRIMARY_VALUE_NONREGRESSION must be YES or NO" >&2
    exit 2
  fi
  TRANSACTION_ARGS=(
    --transactional-main-metric-guard
    --transaction-interval-updates "$TRANSACTION_INTERVAL_UPDATES"
    --transaction-eval-n-envs "$TRANSACTION_EVAL_N_ENVS"
    --transaction-eval-seed "$TRANSACTION_EVAL_SEED"
    --transaction-min-primary-gap-improvement "$TRANSACTION_MIN_PRIMARY_GAP_IMPROVEMENT"
    --transaction-primary-full-rate-tolerance "$TRANSACTION_PRIMARY_FULL_RATE_TOLERANCE"
    "${TRANSACTION_PRIMARY_VALUE_ARGS[@]}"
    --transaction-secondary-gap-tolerance "$TRANSACTION_SECONDARY_GAP_TOLERANCE"
    --transaction-max-arm-qvel-limit-exceed-fraction "$TRANSACTION_MAX_ARM_QVEL_LIMIT_EXCEED_FRACTION"
    --transaction-max-arm-qacc-limit-exceed-fraction "$TRANSACTION_MAX_ARM_QACC_LIMIT_EXCEED_FRACTION"
  )
elif [[ "$TRANSACTIONAL_MAIN_METRIC_GUARD" != NO ]]; then
  echo "TRANSACTIONAL_MAIN_METRIC_GUARD must be YES or NO" >&2
  exit 2
fi

[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
for required_file in \
  "$XML_PATH" \
  "$RMP_OUTPUT_REPLAY_REPORT" \
  "$QREF_SCREEN" \
  "$RESUME_CHECKPOINT"; do
  [[ -f "$required_file" ]] || {
    echo "missing required input: $required_file" >&2
    exit 2
  }
done
[[ "$MAX_STAGES" =~ ^[0-9]+$ ]] || {
  echo "MAX_STAGES must be an integer in 1..34" >&2
  exit 2
}
if ! (( MAX_STAGES >= 1 && MAX_STAGES <= 34 )); then
  echo "MAX_STAGES must be in 1..34" >&2
  exit 2
fi
[[ "$N_ENVS" == 128 || "$N_ENVS" == 512 || "$N_ENVS" == 1024 ]] || {
  echo "N_ENVS must be 128/512 (bounded trial) or 1024 (formal)" >&2
  exit 2
}
if [[ "$N_ENVS" == 128 && "$MINIBATCH_SIZE" != 2048 ]]; then
  echo "N_ENVS=128 requires MINIBATCH_SIZE=2048" >&2
  exit 2
fi
if [[ "$N_ENVS" == 512 && "$MINIBATCH_SIZE" != 8192 ]]; then
  echo "N_ENVS=512 requires MINIBATCH_SIZE=8192" >&2
  exit 2
fi
if [[ "$N_ENVS" == 1024 && "$MINIBATCH_SIZE" != 16384 ]]; then
  echo "N_ENVS=1024 requires MINIBATCH_SIZE=16384" >&2
  exit 2
fi
[[ "$ADVANCE_EVAL_N_ENVS" == 64 || "$ADVANCE_EVAL_N_ENVS" == 128 ]] || {
  echo "ADVANCE_EVAL_N_ENVS must be 64 or 128" >&2
  exit 2
}
[[ "$NEW_BALL_MASS_MIN_KG" == 0.0030 ]] || {
  echo "V37 fixes NEW_BALL_MASS_MIN_KG=0.0030" >&2
  exit 2
}
[[ "$NEW_BALL_MASS_MAX_KG" == 0.0040 ]] || {
  echo "V37 fixes NEW_BALL_MASS_MAX_KG=0.0040" >&2
  exit 2
}
actual_resume_sha256=$(sha256sum "$RESUME_CHECKPOINT" | awk '{print $1}')
[[ "$actual_resume_sha256" == "$EXPECTED_RESUME_SHA256" ]] || {
  echo "resume checkpoint SHA-256 mismatch" >&2
  echo "expected: $EXPECTED_RESUME_SHA256" >&2
  echo "actual:   $actual_resume_sha256" >&2
  exit 2
}
[[ ! -e "$RUN_DIR" ]] || {
  echo "refusing to overwrite output directory: $RUN_DIR" >&2
  exit 2
}

actual_gpu_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$actual_gpu_uuid" == "$EXPECTED_GPU_UUID" ]] || {
  echo "GPU0 UUID mismatch: expected $EXPECTED_GPU_UUID, got $actual_gpu_uuid" >&2
  exit 2
}
gpu0_processes=$(nvidia-smi -i 0 \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader || true)
if [[ -n "$gpu0_processes" ]]; then
  echo "GPU0 already has compute processes; refusing to share it:" >&2
  echo "$gpu0_processes" >&2
  exit 2
fi
available_memory_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_memory_kib < 8388608 )); then
  echo "less than 8 GiB host memory available; refusing to launch" >&2
  exit 2
fi

echo "Source audit:"
git -C "$REPO_ROOT" status --short --branch --untracked-files=no
echo "GPU audit:"
nvidia-smi \
  --query-gpu=index,uuid,name,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader
echo "Input hashes:"
sha256sum \
  "$XML_PATH" \
  "$RMP_OUTPUT_REPLAY_REPORT" \
  "$QREF_SCREEN" \
  "$RESUME_CHECKPOINT" \
  "$RL_SIM/mjx_juggle_env.py" \
  "$RL_SIM/train_juggle_mjx_curriculum.py" \
  "$RL_SIM/run_with_host_memory_guard.sh" \
  "$RL_SIM/run_gpu0_measured_qvel_rmp_vertical_v37_resume_v36.sh" \
  "$LAUNCHER_AUDIT_PATH" \
  "$REPO_ROOT/AGENTS.md" \
  "$REPO_ROOT/TRAINING_CURRICULA.md"
echo "Resume source: $RESUME_CHECKPOINT"
echo "Selection: $SOURCE_SELECTION_AUDIT"
echo "$SOURCE_METRIC_AUDIT"
echo "$SOURCE_STATE_AUDIT"
echo "Stages 1-19: all V36 non-execution DR before wide RMP/PD; execution stays small."
echo "$STAGE_20_AUDIT"
echo "Stages 21-28: V36 RMP micro/full then PD/plant micro/full; other DR stays maximal."
echo "Update limit: max_stage_updates=$MAX_STAGE_UPDATES; stages=$MAX_STAGES."
echo "Parallel environments: $N_ENVS."
echo "Minibatch size: $MINIBATCH_SIZE (eight minibatches per epoch)."
echo "PPO: LR=$LEARNING_RATE, epochs=$UPDATE_EPOCHS, clip=$CLIP_RANGE, target_KL=$TARGET_KL."
echo "Convergence: window=$CONVERGENCE_WINDOW, warmup=$STAGE_METRIC_WARMUP_UPDATES."
echo "Advance validation environments: $ADVANCE_EVAL_N_ENVS."
echo "Clear JAX caches between stages: $CLEAR_JAX_CACHES_BETWEEN_STAGES."
echo "Run directory: $RUN_DIR"
echo "W&B continuation: id=$WANDB_ID resume=$WANDB_RESUME offset=$WANDB_STEP_OFFSET."
echo "Runtime safety: no GPU sharing, no XLA preallocation, stop at 78 C."

if [[ ${CONFIRM_GPU0_READY:-NO} != YES ]]; then
  echo "Preflight only. Inspect the audit, then set CONFIRM_GPU0_READY=YES." >&2
  exit 3
fi
if [[ ${ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE:-NO} != YES ]]; then
  echo "Set ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE=YES for this experimental continuation." >&2
  exit 2
fi

mkdir -p \
  "$RUN_DIR" \
  "$RUN_DIR/matplotlib" \
  "$RUN_DIR/wandb" \
  "$RUN_DIR/wandb_cache" \
  "$RUN_DIR/wandb_config"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MPLCONFIGDIR="$RUN_DIR/matplotlib"
export WANDB_DISABLE_CODE=true
export WANDB_INIT_TIMEOUT=180
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_CACHE_DIR="$RUN_DIR/wandb_cache"
export WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"

cd "$RL_SIM"

exec "$RL_SIM/run_with_host_memory_guard.sh" \
  "$RUN_DIR/stdout_stderr.log" \
  "$PYTHON_BIN" -u train_juggle_mjx_curriculum.py \
  --xml "$XML_PATH" \
  --save-dir "$RUN_DIR" \
  --seed "$RUN_SEED" \
  --curriculum-profile "$PROFILE" \
  --resume-from "$RESUME_CHECKPOINT" \
  --resume-start-stage "$RESUME_STAGE" \
  "${CURRICULUM_RESUME_ARGS[@]}" \
  "${OBS_DIM_MIGRATION_ARGS[@]}" \
  --curriculum-gate-preset v7_strict \
  --delay-ablation-preset baseline_current \
  --measured-ball-mass-range-kg \
    "$NEW_BALL_MASS_MIN_KG" "$NEW_BALL_MASS_MAX_KG" \
  --rmp-output-replay-report "$RMP_OUTPUT_REPLAY_REPORT" \
  --allow-experimental-rmp-dr-bypass \
  --right-arm-pd-profile recovered_rmp_rmpmd_v2 \
  --actuator-compensation-mode none \
  --no-arm-servo-target-tracking-planner \
  --asymmetric-critic \
  --critic-command-history-steps 12 \
  --n-envs "$N_ENVS" \
  --n-steps 128 \
  --minibatch-size "$MINIBATCH_SIZE" \
  --update-epochs "$UPDATE_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --gamma 0.9995 \
  --gae-lambda 0.99 \
  --time-limit-bootstrap \
  --clip-range "$CLIP_RANGE" \
  --target-kl "$TARGET_KL" \
  --ent-coef 0.0002 \
  --min-log-std -4.8 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  "${TRANSACTION_ARGS[@]}" \
  --advance-mode converged \
  --convergence-window "$CONVERGENCE_WINDOW" \
  --convergence-min-episodes 64 \
  --min-stage-updates 30 \
  --stage-metric-warmup-updates "$STAGE_METRIC_WARMUP_UPDATES" \
  --max-stage-updates "$MAX_STAGE_UPDATES" \
  --max-stages "$MAX_STAGES" \
  --advance-validation-mode block \
  "${CLEAR_STAGE_CACHE_ARGS[@]}" \
  --advance-eval-stochastic \
  --advance-eval-n-envs "$ADVANCE_EVAL_N_ENVS" \
  --advance-eval-steps 1200 \
  --advance-eval-min-episodes 64 \
  --advance-eval-retry-updates 20 \
  --advance-eval-hit-ratio 0.50 \
  --advance-eval-len-ratio 0.50 \
  --advance-eval-min-hits 1.0 \
  --advance-eval-min-len-frac 0.10 \
  --advance-eval-min-return -1000000 \
  --advance-eval-hit-rate-margin 0.05 \
  --advance-eval-hit-interval-margin 0.03 \
  --advance-eval-cond-hit-ratio 0.90 \
  --advance-eval-camera-margin 0.05 \
  --advance-eval-ball-view-margin 0.05 \
  --advance-eval-z-ideal-margin 0.05 \
  --advance-eval-reset-bucket-mode cvar \
  --advance-eval-reset-bucket-min-episodes 8 \
  --advance-eval-reset-bucket-cvar-frac 0.25 \
  --advance-eval-reset-bucket-rate-margin 0.05 \
  --advance-eval-reset-bucket-hit-margin 1.0 \
  --save-every-updates 5 \
  --archive-every-updates 50 \
  --gpu-max-temp-c 78 \
  --gpu-check-every-updates 1 \
  --wandb \
  --wandb-mode "$WANDB_MODE" \
  --wandb-metrics-only \
  --wandb-project pingpong-mjx \
  --wandb-id "$WANDB_ID" \
  --wandb-resume "$WANDB_RESUME" \
  --wandb-step-offset "$WANDB_STEP_OFFSET" \
  --wandb-name "$WANDB_NAME" \
  --wandb-tags \
    gpu0 continuation measured-qvel recovered-rmp rmpmd-pd bounded-qref \
    per-joint-horizon18-19ms obs57 critic368 "$PROFILE_TAG" "$RESUME_SOURCE_TAG" \
    nonexecution-first "$POLISH_TAG" delayed-wide-execution-dr \
    structured-rmp-dr complete-dr-privileged-critic unbounded-updates
