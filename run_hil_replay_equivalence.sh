#!/usr/bin/env bash
set -eo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_env="${PINGPONG_HIL_PYTHON:-/home/spirit-ai/miniforge3/envs/pingpong_mjx/bin/python3.10}"

source /opt/ros/humble/setup.bash
source /home/spirit-ai/ros_pkg/movax_interface/install/setup.bash
if [[ -f "${workspace_dir}/install/setup.bash" ]]; then
  source "${workspace_dir}/install/setup.bash"
fi
set -u

export PYTHONPATH="${workspace_dir}/src/pingpong_hil_rl${PYTHONPATH:+:${PYTHONPATH}}"
export ROS_LOG_DIR="${PINGPONG_ROS_LOG_DIR:-/tmp}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLCONFIGDIR="${PINGPONG_MPLCONFIGDIR:-/tmp/pingpong_mplconfig}"

exec "${python_env}" -m pingpong_hil_rl.replay_equivalence "$@"
