#!/usr/bin/env bash
set -eo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_env="${PINGPONG_FLASH_PYTHON:-/home/spirit-ai/miniforge3/envs/pingpong_mjx/bin/python3.10}"
cudnn_dir="/home/spirit-ai/miniforge3/envs/pingpong_mjx/lib/python3.10/site-packages/nvidia/cudnn/lib"

source /opt/ros/humble/setup.bash
source /home/spirit-ai/ros_pkg/movax_interface/install/setup.bash
if [[ -f "${workspace_dir}/install/setup.bash" ]]; then
  source "${workspace_dir}/install/setup.bash"
fi
set -u

export LD_LIBRARY_PATH="${cudnn_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${workspace_dir}/src/pingpong_hil_rl:${workspace_dir}/third_party/FlashSAC${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export ROS_LOG_DIR="${PINGPONG_ROS_LOG_DIR:-/tmp}"

exec "${python_env}" -m pingpong_hil_rl.train_flashsac "$@"
