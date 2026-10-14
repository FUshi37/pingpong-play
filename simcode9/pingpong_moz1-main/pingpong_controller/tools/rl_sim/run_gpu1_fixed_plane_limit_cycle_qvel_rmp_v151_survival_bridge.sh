#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
OUTPUT_ROOT="$REPO_ROOT/pingpong_controller/outputs/rl_sim"
PYTHON_BIN=/home/yangzhe/miniconda3/envs/pingpong/bin/python
PROFILE=fixed_plane_limit_cycle_qvel_rmp_gpu1_v151
XML_PATH="$RL_SIM/moz1_pd_1ms_v98.xml"
EXPECTED_XML_SHA=ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a
EXPECTED_GPU_UUID=GPU-e74458ec-002a-1bac-e0be-d0a5713b661e
RMP_REPORT="$OUTPUT_ROOT/rmp_measured_qvel_design_20260820/replay_datatracer_training_local_v2.json"
MASS_MANIFEST="$OUTPUT_ROOT/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json"
REPLAY_OBS="$OUTPUT_ROOT/v150_gpu1_preflight_20261006_r3/replay_obs_72d.npy"
REPLAY_ACTIONS="$OUTPUT_ROOT/v150_gpu1_preflight_20261006_r3/replay_actions_7d.npy"
VALIDATION_DIR="$OUTPUT_ROOT/v151_gpu1_reward_validation_20261007_r1"
AUDIT_DIR="$OUTPUT_ROOT/v151_gpu1_reward_audit_20261007_r1"
SOURCE_CHECKPOINT=${V151_RESUME_CHECKPOINT:-$OUTPUT_ROOT/fplc_v150_gpu1_centered_position_resume111_20261006_1845/archive_04_fplc_v150_gpu1_s3_third_confirm_update_1700.pkl}
EXPECTED_SOURCE_SHA=${V151_RESUME_SHA:-b76b6c5d2e82202d3d83b276d7d7e9a50771bbc0ebda63dbdbabe42af3f640d2}
SOURCE_CSV=${V151_RESUME_CSV:-}
EXPECTED_CSV_SHA=${V151_RESUME_CSV_SHA:-}
RESUME_STAGE=fplc_v151_gpu1_s4_fourth_entry
SEED=20263616
RUN_KIND=survival_bridge_formal
N_ENVS=1024
MINIBATCH_SIZE=16384
MAX_STAGE_UPDATES=-1
MAX_STAGES=7
if [[ ${1:-} == --probe ]]; then
  RUN_KIND=survival_bridge_probe
  N_ENVS=256
  MINIBATCH_SIZE=8192
  MAX_STAGE_UPDATES=48
  MAX_STAGES=4
fi
[[ $# -eq 0 || ( $# -eq 1 && ( $1 == --preflight || $1 == --probe ) ) ]] || {
  echo 'usage: launcher [--preflight|--probe]' >&2; exit 2;
}
[[ -n "${TMUX:-}" && "$(tmux display-message -p '#S')" == pp_gpu1 ]] || {
  echo 'V151 must run in the existing pp_gpu1 session' >&2; exit 2;
}
RUN_ID=${RUN_ID:-fplc_v151_gpu1_${RUN_KIND}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-$OUTPUT_ROOT/$RUN_ID}
WANDB_MODE=${WANDB_MODE:-offline}
[[ ! -e "$RUN_DIR" ]] || { echo "refusing existing output: $RUN_DIR" >&2; exit 2; }
[[ "$(sha256sum "$XML_PATH" | awk '{print $1}')" == "$EXPECTED_XML_SHA" ]] || {
  echo 'V151 calibrated 1ms XML mismatch' >&2; exit 2;
}
[[ "$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_SOURCE_SHA" ]] || {
  echo 'V151 source checkpoint hash mismatch' >&2; exit 2;
}
[[ "$(sha256sum "$REPLAY_OBS" | awk '{print $1}')" == 793575f1da785c295ed110a29e1d18052f4f9fa5edc14a332079e534fb210bc3 &&
   "$(sha256sum "$REPLAY_ACTIONS" | awk '{print $1}')" == 131253156dd2084f6062fc0846cd14a6b411fb502866f2fdf65a7897dfa72439 ]] || {
  echo 'V151 observation/action replay hash mismatch' >&2; exit 2;
}
[[ -f "$RMP_REPORT" && -f "$MASS_MANIFEST" && -f "$VALIDATION_DIR/acceptance.json" ]] || {
  echo 'missing reviewed V151 evidence' >&2; exit 2;
}
RESUME_ARGS=(--resume-from "$SOURCE_CHECKPOINT" --resume-start-stage "$RESUME_STAGE")
if [[ -n "$SOURCE_CSV" ]]; then
  [[ -n "$EXPECTED_CSV_SHA" && "$(sha256sum "$SOURCE_CSV" | awk '{print $1}')" == "$EXPECTED_CSV_SHA" ]] || {
    echo 'V151 continuation CSV mismatch' >&2; exit 2;
  }
  RESUME_ARGS+=(--resume-curriculum-state)
fi
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 "$PYTHON_BIN" - "$SOURCE_CHECKPOINT" "$SOURCE_CSV" "$RL_SIM" "$VALIDATION_DIR" "$AUDIT_DIR" <<'CHECK'
import csv, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[3])
from train_juggle_mjx_curriculum import build_curriculum, validate_fixed_plane_v151_resume_payload
from mjx_juggle_env import MjxFixedPlaneV98CoarseContactConfig
p=pickle.loads(Path(sys.argv[1]).read_bytes())
validate_fixed_plane_v151_resume_payload(p, requested_start_stage='fplc_v151_gpu1_s4_fourth_entry',
 resume_curriculum_state=bool(sys.argv[2]), reset_optimizer=False,reset_critic=False)
if sys.argv[2]:
 rows=list(csv.DictReader(Path(sys.argv[2]).open()))
 assert int(rows[-1]['global_step'])==p['step'] and int(rows[-1]['stage_update'])==p['stage_update']
 assert rows[-1]['stage_name']==p['stage_name']
cfg=build_curriculum(curriculum_profile='fixed_plane_limit_cycle_qvel_rmp_gpu1_v151',
 measured_ball_mass_range_kg=(.0039,.0041),asymmetric_critic=True,critic_command_history_steps=12,
 right_arm_pd_profile='recovered_rmp_rmpmd_v2')[3].cfg
cfg=MjxFixedPlaneV98CoarseContactConfig(**vars(cfg))
v=Path(sys.argv[4]); accepted=json.loads((v/'acceptance.json').read_text())
assert accepted['cfg_sha256']==hashlib.sha256(repr(vars(cfg)).encode()).hexdigest()
for i in (0,1):
 d=np.load(v/f'population{i}.npz');a,b,z=d['returns'].reshape(3,64);ids=np.r_[1:32,33:64]
 assert (a[ids]>b[ids]).mean()>=.9 and (a[ids]>z[ids]).mean()>=.9
 if i==0:
  old=np.load(Path(sys.argv[5])/'frozen_reward_scan.npz')
  assert np.array_equal(d['hits'],old['hits']) and np.array_equal(d['length'],old['length'])
  assert np.max(np.abs(d['min_z']-old['min_z']))<1e-6 and np.max(np.abs(d['max_z']-old['max_z']))<1e-6
print('V151 full-state source, reward ranking and unchanged physical outcomes verified')
CHECK
[[ "$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" == "$EXPECTED_GPU_UUID" ]] || {
  echo 'GPU1 UUID mismatch' >&2; exit 2;
}
GPU_PIDS=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')
[[ -z "$GPU_PIDS" ]] || { echo "GPU1 occupied: $GPU_PIDS" >&2; exit 2; }
GPU_TEMP_C=$(nvidia-smi -i 1 --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
(( GPU_TEMP_C < 78 )) || { echo 'GPU1 too hot' >&2; exit 2; }
AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
(( AVAILABLE_KIB >= 12 * 1024 * 1024 )) || { echo 'insufficient host available memory' >&2; exit 2; }
DISK_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
(( DISK_KIB >= 5 * 1024 * 1024 )) || { echo 'insufficient output disk space' >&2; exit 2; }
echo "V151 profile=$PROFILE stage=$RESUME_STAGE n_envs=$N_ENVS run=$RUN_DIR GPU=$EXPECTED_GPU_UUID"
if [[ ${1:-} == --preflight ]]; then exit 0; fi
mkdir -p "$RUN_DIR/source_snapshot" "$RUN_DIR/matplotlib" "$RUN_DIR/wandb" "$RUN_DIR/wandb_cache" "$RUN_DIR/wandb_config"
cp "$RL_SIM"/train_juggle_mjx_{curriculum,ppo}.py "$RL_SIM/validate_juggle_mjx_ppo.py" \
 "$RL_SIM/mjx_juggle_env.py" "$RL_SIM/fixed_contact_plane_training.py" "$RL_SIM"/fixed_plane_v*.py \
 "$XML_PATH" "$RUN_DIR/source_snapshot/"
cp "${BASH_SOURCE[0]}" "$RUN_DIR/launcher_snapshot.sh"
sha256sum "$RUN_DIR/source_snapshot"/* "$RUN_DIR/launcher_snapshot.sh" "$SOURCE_CHECKPOINT" \
 "$RMP_REPORT" "$MASS_MANIFEST" "$VALIDATION_DIR"/*.npz "$VALIDATION_DIR/acceptance.json" > "$RUN_DIR/source_manifest.sha256"
git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=no > "$RUN_DIR/tracked_worktree_status.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS='--xla_gpu_deterministic_ops=true'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR="$RUN_DIR/matplotlib" WANDB_DISABLE_CODE=true WANDB_INIT_TIMEOUT=180
export WANDB_DIR="$RUN_DIR/wandb" WANDB_CACHE_DIR="$RUN_DIR/wandb_cache" WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"
cd "$RL_SIM"
CMD=("$PYTHON_BIN" -u train_juggle_mjx_curriculum.py --xml "$XML_PATH" --save-dir "$RUN_DIR" --seed "$SEED"
 --curriculum-profile "$PROFILE" "${RESUME_ARGS[@]}"
 --counterfactual-replay-obs "$REPLAY_OBS" --counterfactual-replay-actions "$REPLAY_ACTIONS"
 --counterfactual-supervision-coef .05 --counterfactual-vxy-weight-mode uniform
 --curriculum-gate-preset v7_strict --delay-ablation-preset baseline_current
 --measured-ball-mass-range-kg .0039 .0041 --ball-mass-measurement-manifest "$MASS_MANIFEST"
 --rmp-output-replay-report "$RMP_REPORT" --allow-experimental-rmp-dr-bypass
 --right-arm-pd-profile recovered_rmp_rmpmd_v2 --actuator-compensation-mode none --no-arm-servo-target-tracking-planner
 --asymmetric-critic --critic-command-history-steps 12
 --n-envs "$N_ENVS" --n-steps 128 --minibatch-size "$MINIBATCH_SIZE" --update-epochs 4
 --learning-rate 3e-5 --gamma .9995 --gae-lambda .99 --time-limit-bootstrap
 --clip-range .12 --target-kl .006 --ent-coef .001 --min-log-std -3.0 --vf-coef .05 --max-grad-norm .5 --hidden-dim 256
 --advance-mode converged --convergence-window 32 --convergence-min-episodes 8 --min-stage-updates 30 --stage-metric-warmup-updates 10
 --max-stage-updates "$MAX_STAGE_UPDATES" --max-stages "$MAX_STAGES"
 --advance-validation-mode block --advance-eval-stochastic --advance-eval-n-envs 288 --advance-eval-steps 1200 --advance-eval-min-episodes 144
 --advance-eval-retry-updates 20 --advance-eval-hit-ratio .50 --advance-eval-len-ratio .50 --advance-eval-min-hits 1.0 --advance-eval-min-len-frac .10
 --advance-eval-min-return -1000000 --advance-eval-hit-rate-margin .05 --advance-eval-hit-interval-margin .03 --advance-eval-cond-hit-ratio .90
 --advance-eval-camera-margin .05 --advance-eval-ball-view-margin .05 --advance-eval-z-ideal-margin .05
 --advance-eval-reset-bucket-mode cvar --advance-eval-reset-bucket-min-episodes 8 --advance-eval-reset-bucket-cvar-frac .25
 --advance-eval-reset-bucket-rate-margin .05 --advance-eval-reset-bucket-hit-margin 1.0
 --save-every-updates 10 --archive-every-updates 100 --gpu-max-temp-c 78 --gpu-check-every-updates 1
 --wandb --wandb-mode "$WANDB_MODE" --wandb-metrics-only --wandb-project pingpong-mjx --wandb-name "$RUN_ID"
 --wandb-tags gpu1 fixed-plane-limit-cycle-v151 reward-survival-bridge position-only-72d camera90hz calibrated-contact no-stochastic-dr)
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
exec "$RL_SIM/run_with_host_memory_guard.sh" "$RUN_DIR/stdout_stderr.log" "${CMD[@]}"
