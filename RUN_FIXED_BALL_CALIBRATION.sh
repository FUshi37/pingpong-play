#!/bin/bash
# Read-only fixed-ball calibration capture. This script does not publish
# /mx_mix_command or any other robot command; arm motion remains manual.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/spirit-ai/pingpong_moz1}"
SETUP_FILE="${SETUP_FILE:-$WORKSPACE_DIR/install/setup.bash}"
COLLECTOR="$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/tools/vision_calib/collect_fixed_ball_bias.py"

if [ -f "$SETUP_FILE" ]; then
    # shellcheck disable=SC1090
    set +u
    source "$SETUP_FILE"
    set -u
fi

export PYTHONPATH="$WORKSPACE_DIR/src/pingpong_moz1${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pingpong_fixed_ball_mplconfig}"
mkdir -p "$MPLCONFIGDIR"

# Match the camera probe launcher: pip CUDA wheels keep cuDNN and related
# shared libraries below site-packages/nvidia/*/lib.
_PY_SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [ -d "$_PY_SITE_PACKAGES/nvidia" ]; then
    _NVIDIA_LD_PATHS=()
    for _NVIDIA_LIB_DIR in "$_PY_SITE_PACKAGES"/nvidia/*/lib; do
        if [ -d "$_NVIDIA_LIB_DIR" ]; then
            _NVIDIA_LD_PATHS+=("$_NVIDIA_LIB_DIR")
        fi
    done
    if [ "${#_NVIDIA_LD_PATHS[@]}" -gt 0 ]; then
        _OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
        IFS=:
        export LD_LIBRARY_PATH="${_NVIDIA_LD_PATHS[*]}${_OLD_LD_LIBRARY_PATH:+:$_OLD_LD_LIBRARY_PATH}"
        unset IFS
    fi
fi

exec python3 -u "$COLLECTOR" "$@"
