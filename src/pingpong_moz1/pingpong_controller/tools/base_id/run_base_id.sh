#!/usr/bin/env bash
# Concise SSH-friendly runner for MOZ1 base dynamics identification (gentle, small-range).
#
# Single-command gentle suite (skips aggressive s3 high-speed brake, max <=0.35 m/s, <=0.25 m):
#   ./src/pingpong_moz1/pingpong_controller/tools/base_id/run_base_id.sh safe       # dry-run preview (no motion)
#   ./src/pingpong_moz1/pingpong_controller/tools/base_id/run_base_id.sh safe_live  # one-shot live run + auto lever-arm + auto fit
#
# Per-stage usage:
#   ./src/pingpong_moz1/pingpong_controller/tools/base_id/run_base_id.sh <stage> [extra args]
#   ./src/pingpong_moz1/pingpong_controller/tools/base_id/run_base_id.sh <stage> live [extra args]
#   ./src/pingpong_moz1/pingpong_controller/tools/base_id/run_base_id.sh fit [run_dirs...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CONTRACT="${SCRIPT_DIR}/base_id_interface_contract.yaml"
SAFE_STAGES=(s0_idle_noise s1_kinematic_low s2_step_ramp_med s4_combined_turn s5_arm_concurrency)

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  $0 safe                                         # dry-run preview of gentle suite (no motion)"
  echo "  $0 safe_live                                    # single-command gentle live suite + auto fit"
  echo "  $0 <stage> [extra --set/--trials args]          # single-stage dry-run (no motion)"
  echo "  $0 <stage> live [extra --set/--trials args]     # single-stage live run"
  echo "  $0 fit [run_dirs...]                            # offline fit & report"
  exit 1
fi

CMD="$1"
shift

_source_ros() {
  set +u
  if [[ -f /opt/ros/humble/setup.bash ]]; then
    source /opt/ros/humble/setup.bash
  fi
  if [[ -f /home/spirit-ai/ros_interface-v2.1.2.3/install/setup.bash ]]; then
    source /home/spirit-ai/ros_interface-v2.1.2.3/install/setup.bash
  fi
  set -u
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-33}"
}

if [[ "${CMD}" == "fit" ]]; then
  if [[ $# -eq 0 ]]; then
    shopt -s nullglob
    RUN_DIRS=("${REPO_ROOT}"/outputs/base_id/base_id_*)
    shopt -u nullglob
    if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
      echo "[ERROR] No run directories found under ${REPO_ROOT}/outputs/base_id/base_id_*" >&2
      exit 1
    fi
  else
    RUN_DIRS=("$@")
  fi
  OUT_DIR="${REPO_ROOT}/outputs/base_id/summary_$(date +%Y%m%d_%H%M%S)"
  MPLCONFIGDIR=/tmp/mpl_base_id python3 "${SCRIPT_DIR}/fit_moz1_base_dynamics.py" \
    "${RUN_DIRS[@]}" --out "${OUT_DIR}"
  echo "[DONE] Fit outputs saved to: ${OUT_DIR}"
  exit 0
fi

cd "${REPO_ROOT}"

if [[ "${CMD}" == "safe" ]]; then
  for st in "${SAFE_STAGES[@]}"; do
    python3 "${SCRIPT_DIR}/run_base_id_experiment.py" \
      --stage "${st}" --contract "${CONTRACT}" --dry-run "$@"
  done
  exit 0
fi

if [[ "${CMD}" == "safe_live" ]]; then
  _source_ros
  SESSION_TAG="$(date +%Y%m%d_%H%M%S)"
  SESSION_ROOT="${REPO_ROOT}/outputs/base_id/session_${SESSION_TAG}"
  mkdir -p "${SESSION_ROOT}"

  echo "===================================================================="
  echo "MOZ1 GENTLE SMALL-RANGE BASE-ID SUITE (s0 -> s1 -> s2 -> s4 -> s5)"
  echo "  Max speed : <= 0.35 m/s linear, <= 0.55 rad/s yaw (skips s3 high)"
  echo "  Max range : <= 0.25 m displacement, 0.60 m geofence"
  echo "  Mocap     : K-rigid-body Y in [0.05, 0.60] m (auto lever-arm after s1)"
  echo "  Safety    : VRMOV + LegWaist impedance checked; Ctrl+C / SSH drop -> zero brake"
  echo "  Output    : ${SESSION_ROOT}"
  echo "===================================================================="
  read -r -p "Confirm VRMOV + LegWaist impedance + E-Stop in hand. Type 'SAFE_LIVE' to start: " CONFIRM
  if [[ "${CONFIRM}" != "SAFE_LIVE" ]]; then
    echo "[ABORT] Confirmation did not match 'SAFE_LIVE'. Zero commands sent." >&2
    exit 3
  fi

  export CONFIRM_MOZ1_BASE_ID_LIVE=YES
  LEVER_OVERRIDE=""
  COLLECTED_DIRS=()

  for st in "${SAFE_STAGES[@]}"; do
    RUN_DIR="${SESSION_ROOT}/${st}"
    echo ""
    echo ">>> Starting stage: ${st} (Ctrl+C to emergency-brake & stop)..."
    sleep 1.5
    EXTRA_ARGS=("$@")
    if [[ -n "${LEVER_OVERRIDE}" ]]; then
      EXTRA_ARGS+=(--set "mocap.base_lever_arm_xy_m=${LEVER_OVERRIDE}")
    fi
    python3 "${SCRIPT_DIR}/run_base_id_experiment.py" \
      --stage "${st}" \
      --contract "${CONTRACT}" \
      --run-dir "${RUN_DIR}" \
      --live \
      "${EXTRA_ARGS[@]}" <<< "${st}"
    COLLECTED_DIRS+=("${RUN_DIR}")

    # Auto-calibrate K-rigid-body lever arm immediately after s1_kinematic_low.
    if [[ "${st}" == "s1_kinematic_low" ]]; then
      S1_FIT="${SESSION_ROOT}/_s1_lever_cal"
      if MPLCONFIGDIR=/tmp/mpl_base_id python3 "${SCRIPT_DIR}/fit_moz1_base_dynamics.py" \
        "${RUN_DIR}" --out "${S1_FIT}" >/dev/null 2>&1; then
        LEVER_OVERRIDE="$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
cal = (d.get("odom_vs_mocap") or {}).get("lever_arm_calibration") or {}
if cal.get("calibrated"):
    xy = cal["lever_arm_xy_m"]
    print(f"[{xy[0]:.5f}, {xy[1]:.5f}]")
' "${S1_FIT}/moz1_base_identified_params_v1.json" || true)"
        if [[ -n "${LEVER_OVERRIDE}" ]]; then
          echo "[MOCAP] Auto-calibrated K-rigid-body lever arm from s1: ${LEVER_OVERRIDE}"
        fi
      fi
    fi
  done

  SUMMARY_DIR="${SESSION_ROOT}/fit_summary"
  echo ""
  echo ">>> Running final offline dynamics fit & report..."
  MPLCONFIGDIR=/tmp/mpl_base_id python3 "${SCRIPT_DIR}/fit_moz1_base_dynamics.py" \
    "${COLLECTED_DIRS[@]}" --out "${SUMMARY_DIR}"
  echo ""
  echo "===================================================================="
  echo "[COMPLETE] All gentle base-ID stages finished cleanly!"
  echo "  Params JSON : ${SUMMARY_DIR}/moz1_base_identified_params_v1.json"
  echo "  Report (CN) : ${SUMMARY_DIR}/BASE_ID_REPORT_CN.md"
  echo "  Summary Fig : ${SUMMARY_DIR}/fig_moz1_base_id_summary.png"
  echo "===================================================================="
  exit 0
fi

_source_ros
MODE_FLAG="--dry-run"
if [[ $# -ge 1 && "$1" == "live" ]]; then
  shift
  export CONFIRM_MOZ1_BASE_ID_LIVE=YES
  MODE_FLAG="--live"
fi

exec python3 "${SCRIPT_DIR}/run_base_id_experiment.py" \
  --stage "${CMD}" \
  --contract "${CONTRACT}" \
  "${MODE_FLAG}" \
  "$@"
