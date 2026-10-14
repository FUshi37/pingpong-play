#!/usr/bin/env bash
set -euo pipefail

RL_SIM=/home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim
MEMORY_HIGH=${V85_MEMORY_HIGH_BYTES:-13958643712}
MEMORY_MAX=${V85_MEMORY_MAX_BYTES:-15032385536}

if [[ ${V85_GPU0_MEMORY_CGROUP_ACTIVE:-NO} != YES ]]; then
  exec systemd-run --user --scope --quiet \
    -p MemoryHigh="$MEMORY_HIGH" \
    -p MemoryMax="$MEMORY_MAX" \
    -p MemorySwapMax=0 \
    -p ManagedOOMPreference=avoid \
    env V85_GPU0_MEMORY_CGROUP_ACTIVE=YES "$0"
fi

exec "$RL_SIM/run_gpu0_measured_qvel_rmp_vertical_v85_resume_v83_stage23_update100.sh"
