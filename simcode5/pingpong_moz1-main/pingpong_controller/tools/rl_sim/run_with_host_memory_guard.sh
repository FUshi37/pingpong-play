#!/usr/bin/env bash
set -uo pipefail

if (( $# < 2 )); then
  echo "usage: $0 LOG_FILE COMMAND [ARGS...]" >&2
  exit 2
fi

LOG_FILE=$1
shift
mkdir -p "$(dirname "$LOG_FILE")"

# Keep enough RAM for the desktop, CUDA pinned allocations, and the other GPU
# trainer.  This guard also runs during XLA compilation, before the Python
# training loop's own temperature/safe-stop checks become active.
MIN_AVAILABLE_KB=$((6 * 1024 * 1024))
MAX_SWAP_USED_KB=$((4 * 1024 * 1024))
CRITICAL_AVAILABLE_KB=$((3 * 1024 * 1024))

"$@" > >(tee -a "$LOG_FILE") 2>&1 &
TRAIN_PID=$!
STOP_REQUESTED=0
STOP_REQUEST_TIME=0

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  MEM_AVAILABLE_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  SWAP_TOTAL_KB=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
  SWAP_FREE_KB=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
  SWAP_USED_KB=$((SWAP_TOTAL_KB - SWAP_FREE_KB))

  if (( STOP_REQUESTED == 0 )) && {
    (( MEM_AVAILABLE_KB < MIN_AVAILABLE_KB )) ||
    (( SWAP_USED_KB > MAX_SWAP_USED_KB ));
  }; then
    echo "[host_memory_guard] safe stop: available=${MEM_AVAILABLE_KB}KiB swap_used=${SWAP_USED_KB}KiB" | tee -a "$LOG_FILE"
    kill -INT "$TRAIN_PID" 2>/dev/null || true
    STOP_REQUESTED=1
    STOP_REQUEST_TIME=$SECONDS
  fi

  if (( STOP_REQUESTED == 1 )) &&
    (( SECONDS - STOP_REQUEST_TIME >= 30 )) &&
    (( MEM_AVAILABLE_KB < CRITICAL_AVAILABLE_KB )); then
    echo "[host_memory_guard] critical memory persisted; terminating pid=${TRAIN_PID}" | tee -a "$LOG_FILE"
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
  fi
  sleep 5
done

wait "$TRAIN_PID"
exit $?
