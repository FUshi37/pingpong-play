#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
SOURCE_RUN="$REPO_ROOT/pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/formal_gpu1_v153_dual_domain_from_v152_stage51_nsteps128_online1_20260901"
SOURCE_CHECKPOINT="$SOURCE_RUN/mjx_curriculum_interrupted.pkl"
SOURCE_PROGRESS="$SOURCE_RUN/curriculum_progress.csv"
ANCHOR_REFERENCE="$SOURCE_RUN/source_snapshot/v146_actor_anchor_reference.pkl"
ANCHOR_REPLAY="$SOURCE_RUN/source_snapshot/v146_stage50_main_actor_anchor_obs_8192.npy"
FRONTIER_ACCEPTANCE="$REPO_ROOT/pingpong_controller/outputs/rl_sim/gpu1_b2b3_rootcause_20260831/v153_b2b3_rootcause_and_preflight_20260901/V153_FRONTIER_ACCEPTANCE.json"
BOUNDED_ACCEPTANCE="$REPO_ROOT/pingpong_controller/outputs/rl_sim/gpu1_b2b3_rootcause_20260831/v153_dual_domain_stage52_53_probe3_from_v152_stage51_nsteps128_offline1_20260901/V153_BOUNDED_ACCEPTANCE.json"
TRAINER_SOURCE="$RL_SIM/train_juggle_mjx_curriculum.py"
PPO_SOURCE="$RL_SIM/train_juggle_mjx_ppo.py"
ENV_SOURCE="$RL_SIM/mjx_juggle_env.py"
SMOKE_SOURCE="$RL_SIM/mjx_smoke.py"
MASS_VALIDATOR_SOURCE="$RL_SIM/ball_mass_measurement.py"
MASS_CONTRACT_SOURCE="$REPO_ROOT/pingpong_controller/outputs/rl_sim/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json"
MEMORY_GUARD_SOURCE="$RL_SIM/run_with_host_memory_guard.sh"
XML_SOURCE="$RL_SIM/moz1_pd.xml"
MESH_SOURCE="$RL_SIM/meshes"
PYTHON=/home/yangzhe/miniconda3/envs/pingpong/bin/python

PROFILE=goal_d455_sport_taskspace_record_new3_sim2real_fixed_base_ball4g_dual_domain_homotopy_v153
RESUME_STAGE=record_new3_sim2real_v153_b2_b3_energy_p015_60hz
RUN_DIR="$REPO_ROOT/pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/formal_gpu1_v153_dual_domain_resume1_stage57_u1_nsteps128_online1_20260901"

EXPECTED_GPU1_UUID=GPU-e74458ec-002a-1bac-e0be-d0a5713b661e
EXPECTED_SOURCE_SHA256=6494393731b00a7c2fe96b2051e87e3b15c2ecf30368b5d38fe008ace478adcc
EXPECTED_PROGRESS_SHA256=4e8ad893e3c99be3f0c57e29617100cf9fcae9113f38986fd819dc2a5c8b0df2
EXPECTED_ANCHOR_REFERENCE_SHA256=2d7498c3769e42a1b254ed0472ba269e8418eaa50c3de951a6d6d1bdb085e6d4
EXPECTED_ANCHOR_REPLAY_SHA256=f6a8557ca5d04d8eeceb8fb6522f3b287a6e2a52d9c0f8cbc1e7e6181dc1f587
EXPECTED_FRONTIER_ACCEPTANCE_SHA256=c34d208da2ec4c5323a04e7c87d4ed6926a66647c1b94e2bfccbd00f72c490f7
EXPECTED_BOUNDED_ACCEPTANCE_SHA256=3af3d86c354eb1fdcb32afb28077a4e12cafc4628b2f6fa2d8d90a6e1c4881cc
EXPECTED_TRAINER_SHA256=390d76daf5a8b1bc37237a3389d945ac29e10372a25d4f5da1a80eaf23b8a3d3
EXPECTED_PPO_SHA256=8a0bebb21c8613676095e52c26e33a7c04d90528eb60dae54edf8c18901380d3
EXPECTED_ENV_SHA256=254eb1a94a1286d7c41fa5e3d9548dfdef41f47bd2d3fe5ea639483f41bfb7c4
EXPECTED_SMOKE_SHA256=fd0c4748039ebb53e022857f3cbf673ef50f0ce2ee910655742bb57ad5ce8f68
EXPECTED_MASS_VALIDATOR_SHA256=c298615d811666e1d1a8702ede56bef49482b8bc9f5eab408c083e3e45de7b88
EXPECTED_MASS_CONTRACT_SHA256=d68dec984413dae93b9d88820b7ed674e15111a7edbd4051d4dafd6871d1883f
EXPECTED_MEMORY_GUARD_SHA256=82b38cf6845501e58b0d37a6c98e34b6dc04f99eaedbc1c6c6e226054e3b8470
EXPECTED_XML_SHA256=7d98f2adfdbad6082be0defcec2dbd0cbbcaf1f0fc06ce45ba424b5b3257cc92
EXPECTED_MESH_TREE_SHA256=f5046e260ae4293675214bb0d9b46bed0ea5c3b5483b7377018a1ae114222851
SOURCE_TRUE_STEP=14750285824
BATCH_STEPS=131072
WANDB_STEP_OFFSET=2000945152
EXPECTED_PREVIOUS_WANDB_STEP=16751230976
EXPECTED_FIRST_WANDB_STEP=16751362048

[[ ${CONFIRM_GPU1_V153_RESUME1:-NO} == YES ]] || {
  echo "set CONFIRM_GPU1_V153_RESUME1=YES after reviewing the exact interrupted checkpoint" >&2
  exit 3
}
CURRENT_TMUX_SESSION=$(tmux display-message -p '#S' 2>/dev/null || true)
[[ "$CURRENT_TMUX_SESSION" == pp_gpu1 ]] || {
  echo "V153 resume1 must be launched inside tmux session pp_gpu1" >&2
  exit 3
}
[[ ! -e "$RUN_DIR" ]] || {
  echo "refusing to reuse V153 resume1 output directory: $RUN_DIR" >&2
  exit 2
}
[[ -f "$SOURCE_CHECKPOINT" && -f "$SOURCE_PROGRESS" ]]
[[ -f "$ANCHOR_REFERENCE" && -f "$ANCHOR_REPLAY" ]]
[[ -f "$FRONTIER_ACCEPTANCE" && -f "$BOUNDED_ACCEPTANCE" ]]

[[ $(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}') == "$EXPECTED_SOURCE_SHA256" ]]
[[ $(sha256sum "$SOURCE_PROGRESS" | awk '{print $1}') == "$EXPECTED_PROGRESS_SHA256" ]]
[[ $(sha256sum "$ANCHOR_REFERENCE" | awk '{print $1}') == "$EXPECTED_ANCHOR_REFERENCE_SHA256" ]]
[[ $(sha256sum "$ANCHOR_REPLAY" | awk '{print $1}') == "$EXPECTED_ANCHOR_REPLAY_SHA256" ]]
[[ $(sha256sum "$FRONTIER_ACCEPTANCE" | awk '{print $1}') == "$EXPECTED_FRONTIER_ACCEPTANCE_SHA256" ]]
[[ $(sha256sum "$BOUNDED_ACCEPTANCE" | awk '{print $1}') == "$EXPECTED_BOUNDED_ACCEPTANCE_SHA256" ]]
[[ $(sha256sum "$TRAINER_SOURCE" | awk '{print $1}') == "$EXPECTED_TRAINER_SHA256" ]]
[[ $(sha256sum "$PPO_SOURCE" | awk '{print $1}') == "$EXPECTED_PPO_SHA256" ]]
[[ $(sha256sum "$ENV_SOURCE" | awk '{print $1}') == "$EXPECTED_ENV_SHA256" ]]
[[ $(sha256sum "$SMOKE_SOURCE" | awk '{print $1}') == "$EXPECTED_SMOKE_SHA256" ]]
[[ $(sha256sum "$MASS_VALIDATOR_SOURCE" | awk '{print $1}') == "$EXPECTED_MASS_VALIDATOR_SHA256" ]]
[[ $(sha256sum "$MASS_CONTRACT_SOURCE" | awk '{print $1}') == "$EXPECTED_MASS_CONTRACT_SHA256" ]]
[[ $(sha256sum "$MEMORY_GUARD_SOURCE" | awk '{print $1}') == "$EXPECTED_MEMORY_GUARD_SHA256" ]]
[[ $(sha256sum "$XML_SOURCE" | awk '{print $1}') == "$EXPECTED_XML_SHA256" ]]
MESH_TREE_SHA256=$(cd "$MESH_SOURCE" && sha256sum -- * | sha256sum | awk '{print $1}')
[[ "$MESH_TREE_SHA256" == "$EXPECTED_MESH_TREE_SHA256" ]]

NEW_FIRST_WANDB_STEP=$((SOURCE_TRUE_STEP + BATCH_STEPS + WANDB_STEP_OFFSET))
[[ "$NEW_FIRST_WANDB_STEP" -eq "$EXPECTED_FIRST_WANDB_STEP" ]]
[[ "$NEW_FIRST_WANDB_STEP" -eq $((EXPECTED_PREVIOUS_WANDB_STEP + BATCH_STEPS)) ]]

GPU1_UUID=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader)
[[ "$GPU1_UUID" == "$EXPECTED_GPU1_UUID" ]]
GPU1_TEMP_C=$(nvidia-smi --id=1 --query-gpu=temperature.gpu --format=csv,noheader,nounits)
[[ "$GPU1_TEMP_C" -le 78 ]]
mapfile -t GPU1_PIDS < <(
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits |
    awk -F', ' -v uuid="$EXPECTED_GPU1_UUID" '$1 == uuid {print $2}'
)
[[ ${#GPU1_PIDS[@]} -eq 0 ]] || {
  echo "GPU1 already owns compute PIDs: ${GPU1_PIDS[*]}" >&2
  exit 2
}

mkdir -p "$RUN_DIR/source_snapshot" "$RUN_DIR/wandb" "$RUN_DIR/wandb_cache" "$RUN_DIR/wandb_config"
FROZEN="$RUN_DIR/source_snapshot"
cp "$SOURCE_CHECKPOINT" "$FROZEN/v153_interrupted_stage57_u1_step14750285824.pkl"
cp "$SOURCE_PROGRESS" "$FROZEN/curriculum_progress.csv"
cp "$ANCHOR_REFERENCE" "$FROZEN/v146_actor_anchor_reference.pkl"
cp "$ANCHOR_REPLAY" "$FROZEN/v146_stage50_main_actor_anchor_obs_8192.npy"
cp "$FRONTIER_ACCEPTANCE" "$FROZEN/V153_FRONTIER_ACCEPTANCE.json"
cp "$BOUNDED_ACCEPTANCE" "$FROZEN/V153_BOUNDED_ACCEPTANCE.json"
cp "$TRAINER_SOURCE" "$FROZEN/train_juggle_mjx_curriculum.py"
cp "$PPO_SOURCE" "$FROZEN/train_juggle_mjx_ppo.py"
cp "$ENV_SOURCE" "$FROZEN/mjx_juggle_env.py"
cp "$SMOKE_SOURCE" "$FROZEN/mjx_smoke.py"
cp "$MASS_VALIDATOR_SOURCE" "$FROZEN/ball_mass_measurement.py"
cp "$MEMORY_GUARD_SOURCE" "$FROZEN/run_with_host_memory_guard.sh"
cp "$XML_SOURCE" "$FROZEN/moz1_pd.xml"
cp -a "$MESH_SOURCE" "$FROZEN/meshes"
cp "$MASS_CONTRACT_SOURCE" "$FROZEN/ball_mass_contract_manifest.json"
cp "${BASH_SOURCE[0]}" "$RUN_DIR/launcher_snapshot.sh"
git -C "$REPO_ROOT" status --short --branch > "$RUN_DIR/git_status.txt"
find "$FROZEN" -type f -print0 | sort -z | xargs -0 sha256sum > "$RUN_DIR/source_sha256.txt"
"$PYTHON" "$FROZEN/ball_mass_measurement.py" \
  "$FROZEN/ball_mass_contract_manifest.json" > "$RUN_DIR/ball_mass_contract.json"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="$FROZEN:$RL_SIM${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLE_CODE=true
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_CACHE_DIR="$RUN_DIR/wandb_cache"
export WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"
export GUARD_MIN_AVAILABLE_KB=$((5 * 1024 * 1024))
export GUARD_CRITICAL_AVAILABLE_KB=$((2 * 1024 * 1024))
export GUARD_SWAP_PRESSURE_AVAILABLE_KB=$((8 * 1024 * 1024))
export GUARD_MAX_SWAP_GROWTH_KB=$((2 * 1024 * 1024))

cd "$FROZEN"
exec "$FROZEN/run_with_host_memory_guard.sh" "$RUN_DIR/stdout_stderr.log" \
  "$PYTHON" -u train_juggle_mjx_curriculum.py \
  --xml "$FROZEN/moz1_pd.xml" \
  --save-dir "$RUN_DIR" \
  --seed 82731 \
  --curriculum-gate-preset legacy \
  --curriculum-profile "$PROFILE" \
  --delay-ablation-preset sport_actuator_record_new3_recalibrated_dr \
  --ball-mass-measurement-manifest "$FROZEN/ball_mass_contract_manifest.json" \
  --resume-from "$FROZEN/v153_interrupted_stage57_u1_step14750285824.pkl" \
  --resume-curriculum-state \
  --resume-start-stage record_new3_sim2real_v153_b2_b3_energy_p015_60hz \
  --no-allow-obs-dim-migration \
  --actor-anchor-reference-checkpoint "$FROZEN/v146_actor_anchor_reference.pkl" \
  --actor-anchor-replay-obs "$FROZEN/v146_stage50_main_actor_anchor_obs_8192.npy" \
  --actor-anchor-replay-kl-coef 0.02 \
  --actor-anchor-replay-max-kl 0.025 \
  --n-envs 1024 --n-steps 128 --minibatch-size 16384 --update-epochs 2 \
  --learning-rate 1e-5 --gamma 0.9995 --gae-lambda 0.995 --time-limit-bootstrap \
  --clip-range 0.06 --target-kl 0.002 --ent-coef 0.0001 \
  --min-log-std -4.2 --max-log-std -3.8 \
  --actor-anchor-kl-coef 0.01 --residual-l2-coef 0.01 \
  --vf-coef 0.25 --max-grad-norm 0.5 \
  --asymmetric-critic --critic-sim2real-privileged --critic-command-history-steps 12 \
  --advance-mode converged --convergence-window 48 --convergence-min-episodes 64 \
  --min-stage-updates 16 --stage-metric-warmup-updates 16 \
  --max-stage-updates -1 --max-stages 66 \
  --stop-after-stage-name record_new3_sim2real_v153_main_racket_launch_combined_dual_domain_proof_60hz \
  --advance-validation-mode block --advance-eval-n-envs 512 \
  --advance-eval-steps 1200 --advance-eval-min-episodes 64 \
  --advance-eval-retry-updates 10 --advance-eval-min-hits 1.5 \
  --advance-eval-min-len-frac 0.10 --advance-eval-min-return -2.0 \
  --advance-eval-hit-ratio 0.90 --advance-eval-len-ratio 0.90 \
  --advance-eval-hit-rate-margin 0.04 --advance-eval-hit-interval-margin 0.04 \
  --advance-eval-cond-hit-ratio 0.90 --advance-eval-camera-margin 0.08 \
  --advance-eval-ball-view-margin 0.04 --advance-eval-z-ideal-margin 0.06 \
  --advance-eval-reset-bucket-mode worst --advance-eval-reset-bucket-min-episodes 16 \
  --advance-eval-reset-bucket-rate-margin 0.15 --advance-eval-reset-bucket-hit-margin 4.5 \
  --advance-eval-final-min-len-frac 0.90 \
  --advance-eval-final-min-ball-view-in-bounds 0.92 \
  --advance-retention-stage-name record_new3_sim2real_v153_main_racket_launch_retention_anchor_replay_60hz \
  --advance-retention-final-stage-name record_new3_sim2real_v153_b3_energy_p100_60hz \
  --advance-retention-hit-ratio 1.0 --advance-retention-len-ratio 1.0 \
  --advance-retention-ball-view-margin 0.01 --advance-retention-hit-rate-margin 0.03 \
  --advance-retention-cond-hit-ratio 0.95 --clear-jax-caches-between-stages \
  --save-every-updates 5 --archive-every-updates 100 \
  --gpu-max-temp-c 78 --gpu-check-every-updates 1 \
  --wandb --wandb-mode online --wandb-metrics-only \
  --wandb-project pingpong-mjx --wandb-id v130g1r1 --wandb-resume must \
  --wandb-step-offset 2000945152 \
  --wandb-name gpu1-record-new3-v153-dual-domain-resume1-online \
  --wandb-tags gpu1 record-new3 v153 resume1 stage57-u1 dual-domain fixed-base ball4g online
