#!/usr/bin/env bash
# --dry-run validates the 50 postures without ROS or robot motion.
# Live mode publishes slow right-arm joint commands on /mx_mix_command.
# Run: ./RUN_AUTO_MOCAP_CALIBRATION.sh
# Slower: ./RUN_AUTO_MOCAP_CALIBRATION.sh --speed-deg-s 3 --settle-seconds 5
# Retries: --max-attempts 50 --retry-seconds 1
# Points that remain unstable or exhaust sampling attempts are logged and skipped.
# Missing samples are replaced by intermediate poses inside the V172 position
# workspace until 50 valid pairs are collected (at most 50 supplemental points).
# Extra budget: --max-supplemental-points 80; disable: --max-supplemental-points 0
# Supplemental orientation perturbations default to <=2 degrees, within 4 degrees
# of the supplemental starting orientation; targets/paths/feedback use FK checks.
# Resume saved pose pairs: ./RUN_AUTO_MOCAP_CALIBRATION.sh --resume
# Sampling uses measured FK after settling; --arrival-mode target adds target tolerance.
# Keep other motion publishers stopped while this script controls the arm.
set -euo pipefail

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"
if [[ -f "$WORKSPACE_DIR/install/setup.bash" ]]; then
    set +u
    source "$WORKSPACE_DIR/install/setup.bash"
    set -u
fi
export PYTHONPATH="$WORKSPACE_DIR/src/pingpong_moz1:$WORKSPACE_DIR/src/pingpong_replay${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec python3 -u -m pingpong_controller.mocap_auto_calibration "$@"
