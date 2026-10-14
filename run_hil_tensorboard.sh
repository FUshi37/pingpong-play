#!/usr/bin/env bash
set -eo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tensorboard_bin="${PINGPONG_TENSORBOARD_BIN:-/home/spirit-ai/miniforge3/envs/pingpong_mjx/bin/tensorboard}"
logdir="${PINGPONG_TENSORBOARD_LOGDIR:-${workspace_dir}/runs/hil_flashsac}"
host="${PINGPONG_TENSORBOARD_HOST:-127.0.0.1}"
port="${PINGPONG_TENSORBOARD_PORT:-6006}"

exec "${tensorboard_bin}" \
  --logdir "${logdir}" \
  --host "${host}" \
  --port "${port}" \
  --reload_interval 5
