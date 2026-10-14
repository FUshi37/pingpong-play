#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && $1 == --formal ]] || exit 2
KIND=${1#--}
REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
OUTPUT_ROOT="$REPO_ROOT/pingpong_controller/outputs/rl_sim"
PYTHON_BIN=/home/yangzhe/miniconda3/envs/pingpong/bin/python
PROFILE=fixed_plane_limit_cycle_qvel_rmp_gpu0_v161
EVIDENCE="$OUTPUT_ROOT/v161_evidence_20261009_r1"
SOURCE=${V161_RESUME_CHECKPOINT:-$OUTPUT_ROOT/fplc_v154_gpu0_small_dr_formal_20261008_r4/mjx_curriculum_interrupted.pkl}
SOURCE_SHA=${V161_RESUME_SHA:-e770d59ede246481ac6e31198c262475e4d832ab099abdc763b7193133bb4410}
GPU_UUID=GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4
XML="$RL_SIM/moz1_pd_1ms_v98.xml"
RUN_ID=${RUN_ID:-fplc_v161_gpu0_mean_quality_${KIND}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ -n "${TMUX:-}" && "$(tmux display-message -p -t "$TMUX_PANE" '#S')" == pp_gpu0 ]] || exit 2
[[ ! -e "$RUN_DIR" ]] || { echo "refusing existing output: $RUN_DIR" >&2; exit 2; }
[[ $(sha256sum "$XML" | awk '{print $1}') == ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a ]] || exit 2
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 "$PYTHON_BIN" "$RL_SIM/preflight_fixed_plane_v161.py" \
  --checkpoint "$SOURCE" --checkpoint-sha "$SOURCE_SHA" --evidence "$EVIDENCE" --kind "$KIND" --gpu1-run-dir "$OUTPUT_ROOT/fplc_v159_gpu1_mean_quality_formal_20261009_r1"
[[ $(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]') == "$GPU_UUID" ]] || exit 2
[[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]') ]] || exit 2
GPU_TEMP=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
DISK_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
(( GPU_TEMP < 78 && AVAILABLE_KIB >= 12*1024*1024 && DISK_KIB >= 5*1024*1024 )) || exit 2
RESUME_ARGS=(--resume-from "$SOURCE" --resume-start-stage fplc_v161_gpu0_s7_clean_stability)
MAX_UPDATES=-1
MAX_STAGES=14
MIN_UPDATES=30
if [[ ${V161_RESUME_CURRICULUM_STATE:-NO} == YES ]]; then
  RESUME_ARGS+=(--resume-curriculum-state)
fi
mkdir -p "$RUN_DIR/matplotlib" "$RUN_DIR/wandb" "$RUN_DIR/wandb_cache" "$RUN_DIR/wandb_config"
cp "$EVIDENCE/reward_analysis.json" "$RUN_DIR/formal_admission.json"
"$PYTHON_BIN" "$RL_SIM/freeze_training_source.py" --repo-root "$REPO_ROOT" --run-dir "$RUN_DIR" --launcher "${BASH_SOURCE[0]}"
git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=no > "$RUN_DIR/tracked_worktree_status.txt"
sha256sum "$SOURCE" "$XML" "$EVIDENCE/reward_analysis.json" "$RUN_DIR/source_snapshot/source_snapshot_manifest.json" > "$RUN_DIR/source_manifest.sha256"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_deterministic_ops=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export MPLCONFIGDIR="$RUN_DIR/matplotlib" WANDB_DISABLE_CODE=true WANDB_INIT_TIMEOUT=180
export WANDB_DIR="$RUN_DIR/wandb" WANDB_CACHE_DIR="$RUN_DIR/wandb_cache" WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"
RMP_REPORT="$OUTPUT_ROOT/rmp_measured_qvel_design_20260820/replay_datatracer_training_local_v2.json"
MASS_MANIFEST="$OUTPUT_ROOT/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json"
cd "$RL_SIM"
CMD=("$PYTHON_BIN" -u train_juggle_mjx_curriculum.py --xml "$XML" --save-dir "$RUN_DIR" --seed "${V161_SEED:-20263621}"
 --curriculum-profile "$PROFILE" "${RESUME_ARGS[@]}"
 --compact-rollout-metrics --reuse-advance-validation-runtime --reexec-between-stages --reexec-from-frozen-source
 --max-in-memory-stage-history-rows 96 --same-stage-reexec-every-updates 500
 --curriculum-gate-preset v7_strict --delay-ablation-preset baseline_current
 --measured-ball-mass-range-kg .0039 .0041 --ball-mass-measurement-manifest "$MASS_MANIFEST"
 --rmp-output-replay-report "$RMP_REPORT" --allow-experimental-rmp-dr-bypass
 --right-arm-pd-profile recovered_rmp_rmpmd_v2 --actuator-compensation-mode none --no-arm-servo-target-tracking-planner
 --asymmetric-critic --critic-command-history-steps 12
 --n-envs 1024 --n-steps 128 --minibatch-size 16384 --update-epochs 6
 --learning-rate 1e-5 --gamma .9995 --gae-lambda .99 --time-limit-bootstrap
 --clip-range .12 --target-kl .006 --ent-coef .001 --min-log-std -3.0 --vf-coef .005 --max-grad-norm .5 --hidden-dim 256
 --advance-mode converged --convergence-window 32 --convergence-min-episodes 8 --min-stage-updates "$MIN_UPDATES" --stage-metric-warmup-updates 10
 --max-stage-updates "$MAX_UPDATES" --max-stages "$MAX_STAGES" --clear-jax-caches-between-stages
 --advance-validation-mode block --advance-eval-stochastic --advance-eval-n-envs 288 --advance-eval-steps 1200 --advance-eval-min-episodes 144
 --advance-eval-retry-updates 20 --advance-eval-hit-ratio 1 --advance-eval-len-ratio 1 --advance-eval-min-hits 1 --advance-eval-min-len-frac .10
 --advance-eval-min-return -1000000 --advance-eval-hit-rate-margin 0 --advance-eval-hit-interval-margin .03 --advance-eval-cond-hit-ratio 1
 --advance-eval-camera-margin 0 --advance-eval-ball-view-margin 0 --advance-eval-z-ideal-margin 0
 --advance-eval-reset-bucket-mode cvar --advance-eval-reset-bucket-min-episodes 8 --advance-eval-reset-bucket-cvar-frac .25
 --advance-eval-reset-bucket-rate-margin .05 --advance-eval-reset-bucket-hit-margin 1
 --advance-retention-stage-name fplc_v161_gpu0_s7_clean_stability
 --save-every-updates 10 --archive-every-updates 100 --gpu-max-temp-c 78 --gpu-check-every-updates 1
 --wandb --wandb-mode offline --wandb-metrics-only --wandb-project pingpong-mjx --wandb-name "$RUN_ID"
 --wandb-tags gpu0 fixed-plane-v161 reward-only motion-center-regularity calibrated-contact)
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
exec "$RL_SIM/run_with_host_memory_guard.sh" "$RUN_DIR/stdout_stderr.log" "${CMD[@]}"
