#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM=$REPO_ROOT/pingpong_controller/tools/rl_sim
SOURCE_RUN_DIR=$REPO_ROOT/pingpong_controller/outputs/rl_sim
SOURCE_RUN_DIR+=/measured_qvel_rmp_vertical_v83_gpu0_seed20260982_20260828_stage21_oom_resume_online1

[[ ${V85_GPU0_MEMORY_CGROUP_ACTIVE:-NO} == YES ]] || {
  echo "V85 must be launched through its zero-swap cgroup wrapper" >&2
  exit 2
}

export SOURCE_RUN_DIR
export RESUME_CHECKPOINT=$SOURCE_RUN_DIR/archive_24_rmp83_nonexecution_inview_wide_xy_stable_consolidation_update_0100.pkl
export EXPECTED_RESUME_SHA256=c7ae82f871211825539412f952a305c3da5a23b9a87ac089fb0087498cf7401f
export RESUME_STAGE=rmp85_nonexecution_inview_wide_xy_stable_consolidation
export PROFILE_OVERRIDE=goal_d455_measured_qvel_rmp_vertical_v85
export PROFILE_TAG=v85
export RESUME_SOURCE_TAG=resume-v83-stage23-update100
export OBS_DIM_MIGRATION_FLAG=--no-allow-obs-dim-migration
export CURRICULUM_RESUME_FLAG=
export MAX_STAGES=32
export MAX_STAGE_UPDATES=${MAX_STAGE_UPDATES:--1}
export N_ENVS=${N_ENVS:-1024}
export MINIBATCH_SIZE=${MINIBATCH_SIZE:-16384}
export RUN_SEED=${RUN_SEED:-20261004}
export LEARNING_RATE=5.0e-5
export UPDATE_EPOCHS=2
export CLIP_RANGE=0.10
export TARGET_KL=0.004
export CONVERGENCE_WINDOW=48
export STAGE_METRIC_WARMUP_UPDATES=16
export ADVANCE_EVAL_N_ENVS=128
export CLEAR_JAX_CACHES_BETWEEN_STAGES=YES
export RUN_ID=${RUN_ID:-measured_qvel_rmp_vertical_v85_gpu0_seed20261004_20260828_stage23_failure_time_survival_online1}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ID=${WANDB_ID:-v72g0s7a1}
export WANDB_RESUME=${WANDB_RESUME:-must}
# V83's last mapped point is 4809555968. Map the first V85 rollout from the
# selected true-step 2866544640 checkpoint to the following 131072-step point.
export WANDB_STEP_OFFSET=${WANDB_STEP_OFFSET:-1943011328}
export WANDB_NAME=${WANDB_NAME:-gpu0-v85-stage23-failure-time-survival}
export LAUNCHER_AUDIT_PATH=$RL_SIM/run_gpu0_measured_qvel_rmp_vertical_v85_resume_v83_stage23_update100.sh
export SOURCE_SELECTION_AUDIT="Resume only V83 Stage-23 update 100 at true step 2866544640 (SHA-256 c7ae82f8..., Adam t=526571), before the later full/conv_len decline."
export SOURCE_METRIC_AUDIT="Frozen same-lane replay: V84 raised full 0.852->0.859 but reduced length 1101.5->1088.2 and moved failed-episode median length 428->244; a sparse horizon event alone does not optimize failure timing."
export SOURCE_STATE_AUDIT="Restore the exact V83 57-D actor, 368-D critic and Adam moments; start a fresh V85 Stage-23 reward-repair window without curriculum-history resume."
export STAGE_20_AUDIT="V85 retains V84's 5000 low-apex weight and 40-point >=13-hit completion event, and adds only a bounded immediate -30*unfinished_fraction penalty on true termination from Stage 23 onward."
export POLISH_TAG=stage23-24-failure-time-survival-repair

export GUARD_MIN_AVAILABLE_KB=4194304
export GUARD_CRITICAL_AVAILABLE_KB=2097152
export GUARD_SWAP_PRESSURE_AVAILABLE_KB=5242880
export GUARD_MAX_SWAP_GROWTH_KB=4194304
export GUARD_LOW_MEMORY_GRACE_CHECKS=3
export GUARD_SWAP_RESERVE_KB=786432

if [[ "$WANDB_MODE" == online ]]; then
  /home/yangzhe/miniconda3/envs/pingpong/bin/wandb login --verify
fi

exec "$RL_SIM/run_gpu0_measured_qvel_rmp_vertical_v37_resume_v36.sh"
