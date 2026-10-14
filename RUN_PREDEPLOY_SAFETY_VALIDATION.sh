#!/usr/bin/env bash
# Read-only/offline by default. Dynamic commands require --builtin-sim.
set -euo pipefail
SAFETY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SAFETY_ROOT/src/pingpong_moz1:$SAFETY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
for safety_arg in "$@"; do
    if [[ "$safety_arg" == --builtin-sim ]]; then
        set +u
        source /opt/ros/humble/setup.bash
        if [[ -f "$SAFETY_ROOT/install/setup.bash" ]]; then
            source "$SAFETY_ROOT/install/setup.bash"
        fi
        set -u
        break
    fi
done
exec python3 "$SAFETY_ROOT/tools/predeploy_safety_validation.py" "$@"
