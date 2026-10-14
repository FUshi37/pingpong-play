#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && $1 == --formal ]] || exit 2
KIND=${1#--}
REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
OUTPUT_ROOT="$REPO_ROOT/pingpong_controller/outputs/rl_sim"
PYTHON_BIN=/home/yangzhe/miniconda3/envs/pingpong/bin/python
PROFILE=fixed_plane_limit_cycle_qvel_rmp_gpu1_v172
EVIDENCE="$OUTPUT_ROOT/v172_evidence_20261009_r1"
SOURCE=${V172_RESUME_CHECKPOINT:-$OUTPUT_ROOT/fplc_v159_gpu1_mean_quality_formal_20261009_r1/mjx_curriculum_interrupted.pkl}
SOURCE_SHA=${V172_RESUME_SHA:-f7a2faeaaa146e8ddc4b27520e012752343899efc7a52e7e5d85ea9f89e3c1a0}
GPU_UUID=GPU-e74458ec-002a-1bac-e0be-d0a5713b661e
XML="$RL_SIM/moz1_pd_1ms_v98.xml"
RUN_ID=${RUN_ID:-fplc_v172_gpu1_quality_gates_${KIND}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ -n "${TMUX:-}" && "$(tmux display-message -p -t "$TMUX_PANE" '#S')" == pp_gpu1 ]] || exit 2
[[ ! -e "$RUN_DIR" ]] || { echo "refusing existing output: $RUN_DIR" >&2; exit 2; }
[[ $(sha256sum "$XML" | awk '{print $1}') == ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a ]] || exit 2
RESUME_STAGE=${V172_RESUME_START_STAGE:-fplc_v159_gpu1_s9_motion_bridge}
PREFLIGHT_RESUME_ARGS=(--resume-start-stage "$RESUME_STAGE")
RESUME_ARGS=(--resume-from "$SOURCE" --resume-start-stage "$RESUME_STAGE")
if [[ ${V172_RESUME_COMPLETED_STAGE:-NO} == YES ]]; then
  PREFLIGHT_RESUME_ARGS+=(--resume-completed-stage)
else
  RESUME_ARGS+=(--resume-curriculum-state)
fi
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 "$PYTHON_BIN" "$RL_SIM/preflight_fixed_plane_v172.py" \
  --checkpoint "$SOURCE" --checkpoint-sha "$SOURCE_SHA" --evidence "$EVIDENCE" --kind "$KIND" "${PREFLIGHT_RESUME_ARGS[@]}"
[[ $(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]') == "$GPU_UUID" ]] || exit 2
[[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]') ]] || exit 2
GPU_TEMP=$(nvidia-smi -i 1 --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
DISK_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
(( GPU_TEMP < 78 && AVAILABLE_KIB >= 12*1024*1024 && DISK_KIB >= 5*1024*1024 )) || exit 2
MAX_UPDATES=-1
MAX_STAGES=17
MIN_UPDATES=30
if [[ $KIND == probe ]]; then
  MAX_UPDATES=${V172_PROBE_UPDATES:-48}
  [[ $MAX_UPDATES =~ ^[1-9][0-9]*$ ]] || exit 2
  MAX_STAGES=8
  MIN_UPDATES=$((MAX_UPDATES+100))
  if [[ ${V172_CONTINUE_PROBE:-NO} == YES ]]; then
    RESUME_ARGS+=(--resume-curriculum-state)
  fi
elif [[ ${V172_RESUME_CURRICULUM_STATE:-NO} == YES ]]; then
  RESUME_ARGS+=(--resume-curriculum-state)
fi
mkdir -p "$RUN_DIR/matplotlib" "$RUN_DIR/wandb" "$RUN_DIR/wandb_cache" "$RUN_DIR/wandb_config"
cp "$EVIDENCE/gate_analysis.json" "$RUN_DIR/formal_admission.json"
"$PYTHON_BIN" "$RL_SIM/freeze_training_source.py" --repo-root "$REPO_ROOT" --run-dir "$RUN_DIR" --launcher "${BASH_SOURCE[0]}"
git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=no > "$RUN_DIR/tracked_worktree_status.txt"
sha256sum "$SOURCE" "$XML" "$EVIDENCE/gate_analysis.json" "$RUN_DIR/source_snapshot/source_snapshot_manifest.json" > "$RUN_DIR/source_manifest.sha256"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS=--xla_gpu_deterministic_ops=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export MPLCONFIGDIR="$RUN_DIR/matplotlib" WANDB_DISABLE_CODE=true WANDB_INIT_TIMEOUT=180
export WANDB_DIR="$RUN_DIR/wandb" WANDB_CACHE_DIR="$RUN_DIR/wandb_cache" WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"
RMP_REPORT="$OUTPUT_ROOT/rmp_measured_qvel_design_20260820/replay_datatracer_training_local_v2.json"
MASS_MANIFEST="$OUTPUT_ROOT/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json"
REPLAY_OBS="$OUTPUT_ROOT/v150_gpu1_preflight_20261006_r3/replay_obs_72d.npy"
REPLAY_ACTIONS="$OUTPUT_ROOT/v150_gpu1_preflight_20261006_r3/replay_actions_7d.npy"
[[ $(sha256sum "$REPLAY_OBS" | awk '{print $1}') == 793575f1da785c295ed110a29e1d18052f4f9fa5edc14a332079e534fb210bc3 ]] || exit 2
[[ $(sha256sum "$REPLAY_ACTIONS" | awk '{print $1}') == 131253156dd2084f6062fc0846cd14a6b411fb502866f2fdf65a7897dfa72439 ]] || exit 2
cd "$RL_SIM"
CMD=("$PYTHON_BIN" -u train_juggle_mjx_curriculum.py --xml "$XML" --save-dir "$RUN_DIR" --seed "${V172_SEED:-20263632}"
 --curriculum-profile "$PROFILE" "${RESUME_ARGS[@]}"
 --compact-rollout-metrics --reuse-advance-validation-runtime --reexec-between-stages --reexec-from-frozen-source
 --max-in-memory-stage-history-rows 96 --same-stage-reexec-every-updates 500
 --counterfactual-replay-obs "$REPLAY_OBS" --counterfactual-replay-actions "$REPLAY_ACTIONS"
 --counterfactual-supervision-coef .05 --counterfactual-vxy-weight-mode uniform
 --curriculum-gate-preset v7_strict --delay-ablation-preset baseline_current
 --measured-ball-mass-range-kg .0039 .0041 --ball-mass-measurement-manifest "$MASS_MANIFEST"
 --rmp-output-replay-report "$RMP_REPORT" --allow-experimental-rmp-dr-bypass
 --right-arm-pd-profile recovered_rmp_rmpmd_v2 --actuator-compensation-mode none --no-arm-servo-target-tracking-planner
 --asymmetric-critic --critic-command-history-steps 12
 --n-envs 1024 --n-steps 128 --minibatch-size 16384 --update-epochs 4
 --learning-rate 1.5e-5 --gamma .9995 --gae-lambda .99 --time-limit-bootstrap
 --clip-range .12 --target-kl .006 --ent-coef .001 --min-log-std -3.0 --vf-coef .05 --max-grad-norm .5 --hidden-dim 256
 --advance-mode converged --convergence-window 32 --convergence-min-episodes 8 --min-stage-updates "$MIN_UPDATES" --stage-metric-warmup-updates 10
 --max-stage-updates "$MAX_UPDATES" --max-stages "$MAX_STAGES" --clear-jax-caches-between-stages
 --advance-validation-mode block --advance-eval-stochastic --advance-eval-n-envs 288 --advance-eval-steps 1200 --advance-eval-min-episodes 144
 --advance-eval-retry-updates 20 --advance-eval-hit-ratio 1 --advance-eval-len-ratio 1 --advance-eval-min-hits 1 --advance-eval-min-len-frac .10
 --advance-eval-min-return -1000000 --advance-eval-hit-rate-margin 0 --advance-eval-hit-interval-margin .03 --advance-eval-cond-hit-ratio 1
 --advance-eval-camera-margin 0 --advance-eval-ball-view-margin 0 --advance-eval-z-ideal-margin 0
 --advance-eval-reset-bucket-mode cvar --advance-eval-reset-bucket-min-episodes 8 --advance-eval-reset-bucket-cvar-frac .25
 --advance-eval-reset-bucket-rate-margin .05 --advance-eval-reset-bucket-hit-margin 1
 --advance-retention-stage-name fplc_v159_gpu1_s9_motion_bridge
 --save-every-updates 10 --archive-every-updates 100 --gpu-max-temp-c 78 --gpu-check-every-updates 1
 --wandb --wandb-mode offline --wandb-metrics-only --wandb-project pingpong-mjx --wandb-name "$RUN_ID"
 --wandb-tags gpu1 fixed-plane-v172 gates-only verified-quality retained-rewards calibrated-contact)
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
exec "$RL_SIM/run_with_host_memory_guard.sh" "$RUN_DIR/stdout_stderr.log" "${CMD[@]}"
