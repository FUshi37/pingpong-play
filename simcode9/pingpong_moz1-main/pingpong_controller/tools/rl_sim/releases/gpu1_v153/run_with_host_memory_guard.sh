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
# Overridable per run: a host that starts with several GiB of stale swap already
# occupied has less absolute headroom left than one starting clean, so a fixed
# growth budget is not equally safe (or equally permissive) across launches.
# MemAvailable stays the real OOM guard; the swap budget only limits thrashing.
MIN_AVAILABLE_KB=${GUARD_MIN_AVAILABLE_KB:-$((4 * 1024 * 1024))}
# A 1024-env MJX trainer uses roughly 9--10 GiB RSS on this 32 GiB host.  With
# both GPU trainers alive, Linux legitimately swaps 2--3 GiB of cold pages
# while still reporting 7--8 GiB MemAvailable.  The former 2 GiB growth budget
# therefore stopped healthy jobs (including the 2026-08-16 GPU0 run at
# available=7.9 GiB) rather than identifying an OOM trajectory.  Keep swap as
# an early-warning signal, but allow the measured dual-trainer working set.
MAX_SWAP_GROWTH_KB=${GUARD_MAX_SWAP_GROWTH_KB:-$((4 * 1024 * 1024))}
CRITICAL_AVAILABLE_KB=${GUARD_CRITICAL_AVAILABLE_KB:-$((2 * 1024 * 1024))}
LOW_MEMORY_GRACE_CHECKS=${GUARD_LOW_MEMORY_GRACE_CHECKS:-3}
SWAP_RESERVE_KB=${GUARD_SWAP_RESERVE_KB:-$((128 * 1024))}
# Linux may retain cold anonymous pages in swap even after MemAvailable has
# recovered.  Do not interrupt a healthy trainer solely because that stale
# swap count crossed its launch-relative budget; treat it as pressure only
# when available RAM is also below this early-warning line.
SWAP_PRESSURE_AVAILABLE_KB=${GUARD_SWAP_PRESSURE_AVAILABLE_KB:-$((6 * 1024 * 1024))}
POLL_INTERVAL_SEC=${GUARD_POLL_INTERVAL_SEC:-5}
if (( LOW_MEMORY_GRACE_CHECKS < 1 )); then
  echo "GUARD_LOW_MEMORY_GRACE_CHECKS must be >= 1" >&2
  exit 2
fi
if (( CRITICAL_AVAILABLE_KB >= MIN_AVAILABLE_KB )); then
  echo "GUARD_CRITICAL_AVAILABLE_KB must be below GUARD_MIN_AVAILABLE_KB" >&2
  exit 2
fi
if (( SWAP_PRESSURE_AVAILABLE_KB < MIN_AVAILABLE_KB )); then
  echo "GUARD_SWAP_PRESSURE_AVAILABLE_KB must be >= GUARD_MIN_AVAILABLE_KB" >&2
  exit 2
fi

# Do not attribute swap that was already occupied before this launcher started
# to the new trainer.  A desktop/session restart can legitimately retain more
# than the old absolute 4 GiB threshold and used to stop a healthy run before
# JAX completed its first update.  Keep a bounded growth budget for this
# process, while retaining the independent available-memory emergency limit.
SWAP_TOTAL_KB=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
SWAP_FREE_KB=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
BASELINE_SWAP_USED_KB=$((SWAP_TOTAL_KB - SWAP_FREE_KB))
SWAP_STOP_KB=$((BASELINE_SWAP_USED_KB + MAX_SWAP_GROWTH_KB))
MAX_SAFE_SWAP_USED_KB=$((SWAP_TOTAL_KB - SWAP_RESERVE_KB))
if (( SWAP_STOP_KB > MAX_SAFE_SWAP_USED_KB )); then
  SWAP_STOP_KB=$MAX_SAFE_SWAP_USED_KB
fi
if (( SWAP_STOP_KB < BASELINE_SWAP_USED_KB )); then
  # If the host already starts inside the reserve, permit no additional swap
  # growth without killing an otherwise healthy process before it allocates.
  SWAP_STOP_KB=$BASELINE_SWAP_USED_KB
fi
echo "[host_memory_guard] baseline_swap=${BASELINE_SWAP_USED_KB}KiB swap_stop=${SWAP_STOP_KB}KiB swap_pressure_available=${SWAP_PRESSURE_AVAILABLE_KB}KiB swap_reserve=${SWAP_RESERVE_KB}KiB min_available=${MIN_AVAILABLE_KB}KiB critical_available=${CRITICAL_AVAILABLE_KB}KiB low_memory_grace_checks=${LOW_MEMORY_GRACE_CHECKS}" | tee -a "$LOG_FILE"

"$@" > >(tee -a "$LOG_FILE") 2>&1 &
TRAIN_PID=$!
STOP_REQUESTED=0
STOP_REQUEST_TIME=0
LOW_MEMORY_CHECKS=0

forward_signal_to_trainer() {
  local signal_name=$1
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "[host_memory_guard] forwarding SIG${signal_name} to trainer pid=${TRAIN_PID}" | tee -a "$LOG_FILE"
    kill "-${signal_name}" "$TRAIN_PID" 2>/dev/null || true
    STOP_REQUESTED=1
    STOP_REQUEST_TIME=$SECONDS
  fi
}

trap 'forward_signal_to_trainer INT' INT
trap 'forward_signal_to_trainer TERM' TERM

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  MEM_AVAILABLE_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  SWAP_TOTAL_KB=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
  SWAP_FREE_KB=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
  SWAP_USED_KB=$((SWAP_TOTAL_KB - SWAP_FREE_KB))

  if (( MEM_AVAILABLE_KB < MIN_AVAILABLE_KB )); then
    LOW_MEMORY_CHECKS=$((LOW_MEMORY_CHECKS + 1))
  else
    LOW_MEMORY_CHECKS=0
  fi

  STOP_CAUSE=""
  if (( MEM_AVAILABLE_KB < CRITICAL_AVAILABLE_KB )); then
    STOP_CAUSE="critical_available"
  elif (( LOW_MEMORY_CHECKS >= LOW_MEMORY_GRACE_CHECKS )); then
    STOP_CAUSE="sustained_low_available"
  elif (( SWAP_USED_KB > SWAP_STOP_KB && MEM_AVAILABLE_KB < SWAP_PRESSURE_AVAILABLE_KB )); then
    STOP_CAUSE="swap_growth_under_pressure"
  fi

  if (( STOP_REQUESTED == 0 )) && [[ -n "$STOP_CAUSE" ]]; then
    echo "[host_memory_guard] safe stop: cause=${STOP_CAUSE} available=${MEM_AVAILABLE_KB}KiB swap_used=${SWAP_USED_KB}KiB" | tee -a "$LOG_FILE"
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
  sleep "$POLL_INTERVAL_SEC"
done

wait "$TRAIN_PID"
exit $?
