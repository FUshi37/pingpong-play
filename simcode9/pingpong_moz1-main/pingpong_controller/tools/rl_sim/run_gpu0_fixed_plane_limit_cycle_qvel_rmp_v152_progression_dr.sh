#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
PYTHON_BIN=${PYTHON_BIN:-/home/yangzhe/miniconda3/envs/pingpong/bin/python}
XML_PATH="$RL_SIM/moz1_pd_1ms_v98.xml"
RMP_REPORT="$REPO_ROOT/pingpong_controller/outputs/rl_sim/rmp_measured_qvel_design_20260820/replay_datatracer_training_local_v2.json"
MASS_MANIFEST="$REPO_ROOT/pingpong_controller/outputs/rl_sim/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json"
PROFILE=fixed_plane_limit_cycle_qvel_rmp_gpu0_v152
EXPECTED_GPU_UUID=GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4
EXPECTED_XML_SHA=ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a
SEED=${SEED_V152:-20263616}
N_ENVS=1024
[[ -n "${TMUX:-}" && "$(tmux display-message -p '#S')" == pp_gpu0 ]] || {
  echo "V152 continuation must run inside pp_gpu0" >&2; exit 2;
}
RUN_KIND=progression_dr
SOURCE_CHECKPOINT="/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/fplc_v149_gpu0_precision_resume128_seed20263616_20261006_024618/mjx_curriculum_interrupted.pkl"
EXPECTED_SOURCE_SHA=308b47da1b29ad370b02cd7ddad0f4883f603d35f0c518c5c378c2045051982d
SOURCE_CSV="/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/fplc_v149_gpu0_precision_resume128_seed20263616_20261006_024618/curriculum_progress.csv"
EXPECTED_CSV_SHA=baf2bda506808bf9d83baa2b689230edd061707d2a8be122052f2187f94effcc
RESUME_STAGE=fplc_v152_gpu0_s5_steady_quality
RUN_ID=${RUN_ID:-fplc_v152_gpu0_${RUN_KIND}_seed${SEED}_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/$RUN_ID}
WANDB_MODE=${WANDB_MODE:-offline}

[[ $# -eq 0 || ( $# -eq 1 && $1 == --preflight ) ]] || {
  echo "usage: $0 [--preflight]" >&2; exit 2;
}
[[ -x "$PYTHON_BIN" && -f "$XML_PATH" && -f "$RMP_REPORT" && -f "$MASS_MANIFEST" && -f "$SOURCE_CHECKPOINT" ]] || {
  echo "missing reviewed V152 continuation source" >&2; exit 2;
}
[[ "$(sha256sum "$XML_PATH" | awk '{print $1}')" == "$EXPECTED_XML_SHA" ]] || {
  echo "V152 calibrated 1 ms XML SHA mismatch" >&2; exit 2;
}
[[ "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo "V152 pinned continuation checkpoint SHA mismatch" >&2; exit 2;
}
[[ "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" && "$(sha256sum "$SOURCE_CSV" | awk '{print $1}')" == "$EXPECTED_CSV_SHA" ]] || {
  echo "V152 exact-history checkpoint/CSV hash mismatch" >&2; exit 2;
}
ADMISSION_DIR="$REPO_ROOT/pingpong_controller/outputs/rl_sim/v152_gpu0_admission_20261007_r3"
DR_SMOKE_DIR="$REPO_ROOT/pingpong_controller/outputs/rl_sim/v152_gpu0_dr_smoke_20261007_r1"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 "$PYTHON_BIN" - "$SOURCE_CHECKPOINT" "$SOURCE_CSV" "$RL_SIM" "$ADMISSION_DIR" "$DR_SMOKE_DIR" <<'CHECK_HISTORY'
from pathlib import Path
import csv,hashlib,json,pickle,sys
sys.path.insert(0,sys.argv[3])
from fixed_plane_v152 import validate_progression_resume_payload
from train_juggle_mjx_curriculum import build_curriculum,convergence_status
from types import SimpleNamespace
from argparse import Namespace
p=pickle.loads(Path(sys.argv[1]).read_bytes());rows=list(csv.DictReader(Path(sys.argv[2]).open()))
validate_progression_resume_payload(p,requested_start_stage='fplc_v152_gpu0_s5_steady_quality',
 resume_curriculum_state=False,reset_optimizer=False,reset_critic=False)
assert int(rows[-1]['stage_update'])==p['stage_update'] and int(rows[-1]['global_step'])==p['step']
a=Path(sys.argv[4]);manifest=json.loads((a/'admission.json').read_text())
assert manifest['source_sha256']==hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
assert manifest['seeds']==[20265211,20265212]
for name,sha in manifest['artifacts'].items():
 assert (a/name).parent==a and hashlib.sha256((a/name).read_bytes()).hexdigest()==sha
for seed in manifest['seeds']:
 d=json.loads((a/f'block_{seed}.json').read_text())
 assert d['advance_eval/deterministic']==0 and d['advance_eval/probe_stage_index']==5
 assert d['advance_eval/passed']>.5 and d['advance_eval/episodes']>=144
 assert d['advance_eval/mean_hits']>=12 and d['advance_eval/episode_truncation_rate']>=.95
 assert d['advance_eval/hit12_rate']>=.90 and d['advance_eval/rapid_steady_band_rate']>=.95
 assert d['advance_eval/rapid_steady_surface_z_abs_error_m']<=.060
course=build_curriculum(curriculum_profile='fixed_plane_limit_cycle_qvel_rmp_gpu0_v152',
 measured_ball_mass_range_kg=(.0039,.0041),asymmetric_critic=True,critic_command_history_steps=12,
 right_arm_pd_profile='recovered_rmp_rmpmd_v2')
cfg=course[3]
smoke=json.loads((Path(sys.argv[5])/'smoke.json').read_text())
assert smoke['cfg_sha256']==hashlib.sha256(repr(vars(course[-1].cfg)).encode()).hexdigest()
assert smoke['held_lane_ticks']>0 and smoke['obs_dim']==156 and smoke['reward_channels']>=190
assert all(x['std']>0 for x in smoke['sampled'].values())
def parsed(v):
 try:return float(v)
 except (ValueError,TypeError):return v
history=[{k:parsed(v) for k,v in row.items()} for row in rows]
status=convergence_status(history,cfg,SimpleNamespace(max_steps=1200),Namespace(**p['args']),p['stage_update'])
assert status['convergence/base_stage_converged']>.5
print('V152 pinned full-state source, recomputed rolling gates and two-seed block admission verified')
CHECK_HISTORY
[[ "$N_ENVS" == 1024 ]] || { echo "V152 requires 1024 environments" >&2; exit 2; }
MINIBATCH_SIZE=16384
if (( N_ENVS * 128 < MINIBATCH_SIZE )); then
  MINIBATCH_SIZE=$(( N_ENVS * 128 ))
fi
MAX_STAGE_UPDATES=-1
MAX_STAGES=16
[[ ! -e "$RUN_DIR" ]] || { echo "refusing existing run directory: $RUN_DIR" >&2; exit 2; }
[[ "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" == "$EXPECTED_GPU_UUID" ]] || {
  echo "GPU0 UUID mismatch" >&2; exit 2;
}
GPU_MEM_MIB=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
GPU_TEMP_C=$(nvidia-smi -i 0 --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
(( GPU_MEM_MIB < 2000 && GPU_TEMP_C < 78 )) || {
  echo "GPU0 occupied or too hot: ${GPU_MEM_MIB} MiB, ${GPU_TEMP_C} C" >&2; exit 2;
}
GPU_COMPUTE_PIDS=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' || true)
if [[ -n "$GPU_COMPUTE_PIDS" ]]; then
  while IFS= read -r GPU_PID; do
    [[ -n "$GPU_PID" ]] || continue
    GPU_COMMAND=$(ps -p "$GPU_PID" -o args= | sed 's/^[[:space:]]*//')
    [[ "$GPU_COMMAND" == /usr/bin/totem\ --gapplication-service ]] || {
      echo "GPU0 compute process present: $GPU_PID $GPU_COMMAND" >&2; exit 2;
    }
  done < <(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  echo "GPU0 video viewer is active; resource headroom passed, continuing without closing it."
fi
AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
(( AVAILABLE_KIB >= 12 * 1024 * 1024 )) || { echo "insufficient host available memory" >&2; exit 2; }
DISK_AVAIL_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
(( DISK_AVAIL_KIB >= 5 * 1024 * 1024 )) || { echo "insufficient output disk space" >&2; exit 2; }

echo "V152 GPU0 formal progression audit: profile=$PROFILE seed=$SEED n_envs=$N_ENVS minibatch=$MINIBATCH_SIZE max_stage_updates=$MAX_STAGE_UPDATES"
echo "xml=$XML_PATH xml_sha=$EXPECTED_XML_SHA"
echo "gpu_uuid=$EXPECTED_GPU_UUID gpu_mem_mib=$GPU_MEM_MIB gpu_temp_c=$GPU_TEMP_C"
echo "run_dir=$RUN_DIR wandb_mode=$WANDB_MODE xla_flags=--xla_gpu_deterministic_ops=true"
echo "V152 restores validated V149u2299 actor/critic/Adam, starts fresh named Stage5 history, contact-height gate60mm, nine appended gradual DR lessons, no forced progression or update cap."
echo "source=$SOURCE_CHECKPOINT source_sha=$EXPECTED_SOURCE_SHA start_stage=$RESUME_STAGE admission=$ADMISSION_DIR"
if [[ ${1:-} == --preflight ]]; then exit 0; fi

mkdir -p "$RUN_DIR" "$RUN_DIR/matplotlib" "$RUN_DIR/wandb" "$RUN_DIR/wandb_cache" "$RUN_DIR/wandb_config" "$RUN_DIR/source_snapshot"
cp "$RL_SIM/train_juggle_mjx_curriculum.py" "$RL_SIM/train_juggle_mjx_ppo.py" "$RL_SIM/validate_juggle_mjx_ppo.py" \
  "$RL_SIM/mjx_juggle_env.py" "$RL_SIM/fixed_contact_plane_training.py" "$RL_SIM/fixed_plane_v96.py" "$RL_SIM/fixed_plane_v95.py" "$RL_SIM/fixed_plane_v94.py" \
  "$RL_SIM/fixed_plane_v152.py" "$RL_SIM/fixed_plane_v149.py" "$RL_SIM/fixed_plane_v99.py" "$RL_SIM/fixed_plane_v93.py" "$RL_SIM/fixed_plane_v92.py" "$RL_SIM/fixed_plane_v90.py" "$RL_SIM/fixed_plane_v80.py" \
  "$RL_SIM/fixed_plane_v75.py" "$RL_SIM/fixed_plane_v115.py" "$RL_SIM/fixed_plane_v120.py" "$RL_SIM/fixed_plane_v122.py" "$RL_SIM/fixed_plane_v123.py" "$RL_SIM/fixed_plane_v124.py" "$RL_SIM/fixed_plane_v106.py" "$RL_SIM/fixed_plane_v107.py" "$RL_SIM/fixed_plane_v108.py" "$RL_SIM/fixed_plane_v110.py" "$RL_SIM/fixed_plane_v53.py" "$RL_SIM/fixed_plane_v41.py" \
  "$RL_SIM/fixed_plane_v36.py" "$RL_SIM/rl_juggle_env_random.py" "$RL_SIM/moz1_pd_1ms_v98.xml" \
  "$RUN_DIR/source_snapshot/"
cp "${BASH_SOURCE[0]}" "$RUN_DIR/launcher_snapshot.sh"
sha256sum "$RUN_DIR/source_snapshot"/* "$RUN_DIR/launcher_snapshot.sh" \
  "$RMP_REPORT" "$MASS_MANIFEST" "$SOURCE_CHECKPOINT" "$ADMISSION_DIR"/*.json "$DR_SMOKE_DIR/smoke.json" > "$RUN_DIR/source_manifest.sha256"
git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=no > "$RUN_DIR/tracked_worktree_status.txt"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$RUN_DIR/matplotlib" WANDB_DISABLE_CODE=true WANDB_INIT_TIMEOUT=180
export WANDB_DIR="$RUN_DIR/wandb" WANDB_CACHE_DIR="$RUN_DIR/wandb_cache" WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"
cd "$RL_SIM"
CMD=("$PYTHON_BIN" -u train_juggle_mjx_curriculum.py
  --xml "$XML_PATH" --save-dir "$RUN_DIR" --seed "$SEED"
  --curriculum-profile "$PROFILE"
  --resume-from "$SOURCE_CHECKPOINT" --resume-start-stage "$RESUME_STAGE"
  --curriculum-gate-preset v7_strict --delay-ablation-preset baseline_current
  --measured-ball-mass-range-kg 0.0039 0.0041
  --ball-mass-measurement-manifest "$MASS_MANIFEST" --rmp-output-replay-report "$RMP_REPORT"
  --allow-experimental-rmp-dr-bypass --right-arm-pd-profile recovered_rmp_rmpmd_v2
  --actuator-compensation-mode none --no-arm-servo-target-tracking-planner
  --asymmetric-critic --critic-command-history-steps 12
  --n-envs "$N_ENVS" --n-steps 128 --minibatch-size "$MINIBATCH_SIZE" --update-epochs 6
  --learning-rate 1e-5 --gamma 0.9995 --gae-lambda 0.99 --time-limit-bootstrap
  --clip-range 0.12 --target-kl 0.006 --ent-coef 0.001 --min-log-std -3.0
  --vf-coef 0.005 --max-grad-norm 0.5 --hidden-dim 256
  --advance-mode converged --convergence-window 32 --convergence-min-episodes 8
  --min-stage-updates 30 --stage-metric-warmup-updates 10
  --max-stage-updates "$MAX_STAGE_UPDATES" --max-stages "$MAX_STAGES"
  --clear-jax-caches-between-stages --advance-validation-mode block --advance-eval-stochastic
  --advance-eval-n-envs 288 --advance-eval-steps 1200 --advance-eval-min-episodes 144
  --advance-eval-retry-updates 20 --advance-eval-hit-ratio 0.50 --advance-eval-len-ratio 0.50
  --advance-eval-min-hits 1.0 --advance-eval-min-len-frac 0.10
  --advance-eval-min-return -1000000 --advance-eval-hit-rate-margin 0.05
  --advance-eval-hit-interval-margin 0.03 --advance-eval-cond-hit-ratio 0.90
  --advance-eval-camera-margin 0.05 --advance-eval-ball-view-margin 0.05
  --advance-eval-z-ideal-margin 0.05 --advance-eval-reset-bucket-mode cvar
  --advance-eval-reset-bucket-min-episodes 8 --advance-eval-reset-bucket-cvar-frac 0.25
  --advance-eval-reset-bucket-rate-margin 0.05 --advance-eval-reset-bucket-hit-margin 1.0
  --save-every-updates 10 --archive-every-updates 100
  --gpu-max-temp-c 78 --gpu-check-every-updates 1
  --wandb --wandb-mode "$WANDB_MODE" --wandb-metrics-only
  --wandb-project pingpong-mjx --wandb-name "$RUN_ID"
  --wandb-tags gpu0 fixed-plane-limit-cycle-v152 calibrated-contact direct-release v152-progression-dr gate-aligned-flatness qacc-soft-margin balanced-z-recovery terminal-ranking hit-floor30 widened-z post-hit-quiet signed-height
    calibrated4g staged-candidate-dr measured-qvel bounded-qref recovered-rmp)
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
exec "$RL_SIM/run_with_host_memory_guard.sh" "$RUN_DIR/stdout_stderr.log" "${CMD[@]}"
