#!/usr/bin/env bash
# --dry-run validates the 50 postures without ROS or robot motion.
# Live mode publishes slow right-arm joint commands on /mx_mix_command.
# Run: ./RUN_AUTO_MOCAP_CALIBRATION.sh
# Slower: ./RUN_AUTO_MOCAP_CALIBRATION.sh --speed-deg-s 3 --settle-seconds 5
# Raw mocap rigid-body range (meters, before calibration; blank axis is free):
# --mocap-observation-x-limits-m=-0.4,0.4 --mocap-observation-y-limits-m=0.5,1.5
# --mocap-observation-z-limits-m=0.2,2.0
# Also accepts MOCAP_OBSERVATION_X/Y/Z_LIMITS_M environment variables.
# Range mode selects one in-range rigid body per packet; UDP slot/id may change.
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
