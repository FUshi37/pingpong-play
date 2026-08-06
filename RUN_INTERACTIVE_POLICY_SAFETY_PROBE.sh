#!/bin/bash
# Launch a local browser-based policy safety probe. The Python backend loads
# the MJX deployment policy and simulates the high-level safety gates.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/spirit-ai/pingpong_moz1}"
PROBE_SCRIPT="$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/tools/interactive_policy_safety_probe.py"
SETUP_FILE="${SETUP_FILE:-$WORKSPACE_DIR/install/setup.bash}"
CONFIG_DONE_MONITORING_YAML="${CONFIG_DONE_MONITORING_YAML:-$WORKSPACE_DIR/config_done_monitoring.yaml}"
LOAD_CONFIG_DONE_MONITORING="${LOAD_CONFIG_DONE_MONITORING:-true}"
_DONE_CONFIG_LOADER="$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/tools/load_done_monitoring_env.py"
_YAML_EFFECTIVE_KEYS=()

# Interactive probe launch profile defaults. Seed these before optional YAML
# loading so running this script as-is uses this profile; explicit environment
# variables still win because these assignments only fill unset/empty values.
DONE_PREDICTION_HORIZON_S="${DONE_PREDICTION_HORIZON_S:-0.04}"
DONE_PREDICTION_DT_S="${DONE_PREDICTION_DT_S:-0.005}"
DONE_PREDICTION_MARGIN_M="${DONE_PREDICTION_MARGIN_M:-0.02}"
DONE_PREDICTION_ORIENTATION_MARGIN_RAD="${DONE_PREDICTION_ORIENTATION_MARGIN_RAD:-0.01}"
WORKSPACE_SAFETY_MODE="${WORKSPACE_SAFETY_MODE:-position_soft}"
ENABLE_BALL_STATE_ESTIMATOR="${ENABLE_BALL_STATE_ESTIMATOR:-false}"
RESET_INITIAL_POSE="${RESET_INITIAL_POSE:-default_2_horizontal}"

# Load supported safety parameters before applying the remaining built-in
# defaults below. For variables not seeded by the launch profile above,
# precedence is: explicit environment > YAML > hard-coded fallback.
if [ "$LOAD_CONFIG_DONE_MONITORING" = "true" ]; then
    if [ ! -f "$CONFIG_DONE_MONITORING_YAML" ]; then
        echo "ERROR: done-monitoring config not found: $CONFIG_DONE_MONITORING_YAML" >&2
        echo "Set LOAD_CONFIG_DONE_MONITORING=false to use script defaults." >&2
        exit 2
    fi
    if [ ! -f "$_DONE_CONFIG_LOADER" ]; then
        echo "ERROR: YAML config loader not found: $_DONE_CONFIG_LOADER" >&2
        exit 2
    fi
    if ! _YAML_ENV_LINES="$(
        python3 "$_DONE_CONFIG_LOADER" "$CONFIG_DONE_MONITORING_YAML"
    )"; then
        echo "ERROR: failed to load $CONFIG_DONE_MONITORING_YAML" >&2
        exit 2
    fi
    while IFS=$'\t' read -r _ENV_NAME _ENV_VALUE; do
        [ -n "$_ENV_NAME" ] || continue
        if [[ ! "$_ENV_NAME" =~ ^[A-Z][A-Z0-9_]*$ ]]; then
            echo "ERROR: invalid environment key from YAML loader: $_ENV_NAME" >&2
            exit 2
        fi
        if [[ ! -v $_ENV_NAME ]]; then
            printf -v "$_ENV_NAME" '%s' "$_ENV_VALUE"
            _YAML_EFFECTIVE_KEYS+=("$_ENV_NAME")
        fi
    done <<< "$_YAML_ENV_LINES"
fi

MODEL_PATH="${MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/inverse_compensation/18_stage4g_strong_contact_dr_inverse_mpc.pkl}"
POLICY_PROFILE="${POLICY_PROFILE:-simcode5_pure_actuator67}"
SIMCODE2_MODEL_PATH="${SIMCODE2_MODEL_PATH:-$WORKSPACE_DIR/simcode2/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/goal_d455_autolaunch_idealpd67_viewdense_v1_seed976_gpu1_resume_launch14_best55_20260722/launch19_best2040_video_gpu1_seed20260722/policy_launch19_best_u2040_frozen.pkl}"
SIMCODE2_POLICY_XML_PATH="${SIMCODE2_POLICY_XML_PATH:-$WORKSPACE_DIR/simcode2/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
SIMCODE3_MODEL_PATH="${SIMCODE3_MODEL_PATH:-$WORKSPACE_DIR/simcode3/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/goal_d455_gpu1_resume8_launch19_survival_lr1e4_20260727/mjx_curriculum_best.pkl}"
SIMCODE3_POLICY_XML_PATH="${SIMCODE3_POLICY_XML_PATH:-$WORKSPACE_DIR/simcode3/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
SIMCODE4_MODEL_PATH="${SIMCODE4_MODEL_PATH:-$WORKSPACE_DIR/simcode4/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/mjx_curriculum_best.pkl}"
SIMCODE4_POLICY_XML_PATH="${SIMCODE4_POLICY_XML_PATH:-$WORKSPACE_DIR/simcode4/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
SIMCODE4_REAL_SERVO_PLANNER="${SIMCODE4_REAL_SERVO_PLANNER:-false}"
SIMCODE5_MODEL_PATH="${SIMCODE5_MODEL_PATH:-$WORKSPACE_DIR/simcode5/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl}"
SIMCODE5_POLICY_XML_PATH="${SIMCODE5_POLICY_XML_PATH:-$WORKSPACE_DIR/simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
ROBOT_XML="${ROBOT_XML:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml}"
POLICY_OBS_DIM="${POLICY_OBS_DIM:-67}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

DONE_Z_MIN_M="${DONE_Z_MIN_M:-0.3}"
DONE_Z_MAX_M="${DONE_Z_MAX_M:-1.9}"
DONE_X_RANGE_M="${DONE_X_RANGE_M:-0.5}"
DONE_Y_RANGE_M="${DONE_Y_RANGE_M:-0.5}"
DONE_BALL_REQUIRED_CONSECUTIVE_TICKS="${DONE_BALL_REQUIRED_CONSECUTIVE_TICKS:-5}"
DONE_RACKET_ANCHOR_DISTANCE_M="${DONE_RACKET_ANCHOR_DISTANCE_M:-1.1}"
DONE_RACKET_Z_HIGH_OFFSET_M="${DONE_RACKET_Z_HIGH_OFFSET_M:-0.24}"
DONE_RACKET_Z_LOW_OFFSET_M="${DONE_RACKET_Z_LOW_OFFSET_M:--0.12}"
# Use the no-colon default form so an explicit empty value (or YAML []) keeps
# its documented meaning of disabling that Cartesian limit.
DONE_EE_X_LIMITS_M="${DONE_EE_X_LIMITS_M--0.298,0.163}"
DONE_EE_Y_LIMITS_M="${DONE_EE_Y_LIMITS_M--0.502,-0.290}"
DONE_EE_Z_LIMITS_M="${DONE_EE_Z_LIMITS_M-0.692,1.462}"
DONE_EE_PSI_LIMITS_RAD="${DONE_EE_PSI_LIMITS_RAD:-}"
DONE_EE_B_LIMITS_RAD="${DONE_EE_B_LIMITS_RAD:-}"
DONE_EE_B_MIN_RAD="${DONE_EE_B_MIN_RAD:-}"
DONE_EE_B_MAX_RAD="${DONE_EE_B_MAX_RAD:-}"
DONE_EE_C_LIMITS_RAD="${DONE_EE_C_LIMITS_RAD:-}"
DONE_EE_C_MIN_RAD="${DONE_EE_C_MIN_RAD:-}"
DONE_EE_C_MAX_RAD="${DONE_EE_C_MAX_RAD:-}"
DONE_EE_ORIENTATION_REFERENCE_XYZW="${DONE_EE_ORIENTATION_REFERENCE_XYZW:-}"
DONE_EE_MAX_ORIENTATION_ERROR_DEG="${DONE_EE_MAX_ORIENTATION_ERROR_DEG:-0.0}"
DONE_ENABLE_PREDICTIVE_STOP="${DONE_ENABLE_PREDICTIVE_STOP:-true}"
DONE_PREDICTION_HORIZON_S="${DONE_PREDICTION_HORIZON_S:-0.04}"
DONE_PREDICTION_DT_S="${DONE_PREDICTION_DT_S:-0.005}"
DONE_PREDICTION_MARGIN_M="${DONE_PREDICTION_MARGIN_M:-0.02}"
DONE_PREDICTION_ORIENTATION_MARGIN_RAD="${DONE_PREDICTION_ORIENTATION_MARGIN_RAD:-0.01}"
BALL_LOST_FAILSAFE_TIMEOUT_S="${BALL_LOST_FAILSAFE_TIMEOUT_S:-0.6}"
ENABLE_BALL_CAPTURE_GATE="${ENABLE_BALL_CAPTURE_GATE:-true}"
ENABLE_BALL_RELEASE_GATE="${ENABLE_BALL_RELEASE_GATE:-true}"
ENABLE_BALL_LOST_FAILSAFE="${ENABLE_BALL_LOST_FAILSAFE:-true}"
ENABLE_DONE_MONITORING="${ENABLE_DONE_MONITORING:-true}"
WORKSPACE_SAFETY_MODE="${WORKSPACE_SAFETY_MODE:-position_soft}"
WORKSPACE_SOFT_MARGIN_M="${WORKSPACE_SOFT_MARGIN_M:-0.05}"
WORKSPACE_SOFT_PULL_SPEED_M_S="${WORKSPACE_SOFT_PULL_SPEED_M_S:-0.08}"
WORKSPACE_SOFT_DAMPING="${WORKSPACE_SOFT_DAMPING:-1.0}"
WORKSPACE_SOFT_MAX_CORRECTION_SPEED_M_S="${WORKSPACE_SOFT_MAX_CORRECTION_SPEED_M_S:-0.30}"
WORKSPACE_SOFT_DLS_DAMPING="${WORKSPACE_SOFT_DLS_DAMPING:-0.03}"
WORKSPACE_SOFT_NULLSPACE_DAMPING="${WORKSPACE_SOFT_NULLSPACE_DAMPING:-0.0}"
WORKSPACE_SOFT_POSTURE_GAIN_S="${WORKSPACE_SOFT_POSTURE_GAIN_S:-0.0}"
WORKSPACE_SOFT_BOUNDARY_JOINT_SPEED_SCALE="${WORKSPACE_SOFT_BOUNDARY_JOINT_SPEED_SCALE:-1.0}"
WORKSPACE_HOLD_TRIGGER_MARGIN_M="${WORKSPACE_HOLD_TRIGGER_MARGIN_M:-0.003}"
BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S="${BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S:-0.15}"
BALL_RELEASE_MIN_Z_DROP_M="${BALL_RELEASE_MIN_Z_DROP_M:-0.015}"
BALL_RELEASE_WINDOW_S="${BALL_RELEASE_WINDOW_S:-0.12}"
BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS="${BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS:-3}"
BALL_RELEASE_MAX_OBS_AGE_S="${BALL_RELEASE_MAX_OBS_AGE_S:-0.08}"
BALL_NOISE_STD_M="${BALL_NOISE_STD_M:-0.0}"
CONTROL_RATE_HZ="${CONTROL_RATE_HZ:-200.0}"
# Shared racket-MPC and phase-2 autonomous-juggling defaults; both modes opt-in.
DEFAULT_CONTROL_MODE="${DEFAULT_CONTROL_MODE:-policy}"
RACKET_TRACKING_HEIGHT_OFFSET_M="${RACKET_TRACKING_HEIGHT_OFFSET_M:-0.25}"
RACKET_TRACKING_USE_MPC="${RACKET_TRACKING_USE_MPC:-false}"
RACKET_TRACKING_POSITION_GAIN_S="${RACKET_TRACKING_POSITION_GAIN_S:-4.0}"
RACKET_TRACKING_DLS_DAMPING="${RACKET_TRACKING_DLS_DAMPING:-0.04}"
RACKET_TRACKING_MAX_CARTESIAN_SPEED_M_S="${RACKET_TRACKING_MAX_CARTESIAN_SPEED_M_S:-0.65}"
RACKET_TRACKING_MAX_CARTESIAN_ACCELERATION_M_S2="${RACKET_TRACKING_MAX_CARTESIAN_ACCELERATION_M_S2:-16.0}"
RACKET_TRACKING_JOINT_VELOCITY_SCALE="${RACKET_TRACKING_JOINT_VELOCITY_SCALE:-0.25}"
RACKET_TRACKING_JOINT_ACCELERATION_SCALE="${RACKET_TRACKING_JOINT_ACCELERATION_SCALE:-0.35}"
RACKET_TRACKING_POSTURE_GAIN_S="${RACKET_TRACKING_POSTURE_GAIN_S:-0.12}"
RACKET_TRACKING_MPC_HORIZON_STEPS="${RACKET_TRACKING_MPC_HORIZON_STEPS:-24}"
RACKET_TRACKING_MPC_POSITION_WEIGHT="${RACKET_TRACKING_MPC_POSITION_WEIGHT:-1.0}"
RACKET_TRACKING_MPC_TARGET_VELOCITY_WEIGHT="${RACKET_TRACKING_MPC_TARGET_VELOCITY_WEIGHT:-1.0}"
RACKET_TRACKING_MPC_VELOCITY_WEIGHT="${RACKET_TRACKING_MPC_VELOCITY_WEIGHT:-0.0001}"
RACKET_TRACKING_MPC_SMOOTH_WEIGHT="${RACKET_TRACKING_MPC_SMOOTH_WEIGHT:-0.0002}"
RACKET_TRACKING_MPC_POSTURE_WEIGHT="${RACKET_TRACKING_MPC_POSTURE_WEIGHT:-0.0001}"
RACKET_TRACKING_TARGET_BOUNDARY_MARGIN_M="${RACKET_TRACKING_TARGET_BOUNDARY_MARGIN_M:-0.005}"
RACKET_TRACKING_MAX_BALL_AGE_S="${RACKET_TRACKING_MAX_BALL_AGE_S:-0.12}"
JUGGLE_COUNTDOWN_S="${JUGGLE_COUNTDOWN_S:-3.0}"
JUGGLE_TARGET_APEX_HEIGHT_M="${JUGGLE_TARGET_APEX_HEIGHT_M:-0.10}"
JUGGLE_SEED_APEX_HEIGHT_M="${JUGGLE_SEED_APEX_HEIGHT_M:-0.01}"
JUGGLE_APEX_RAMP_PER_BOUNCE_M="${JUGGLE_APEX_RAMP_PER_BOUNCE_M:-0.025}"
JUGGLE_LAUNCH_PEAK_ACCELERATION_M_S2="${JUGGLE_LAUNCH_PEAK_ACCELERATION_M_S2:-16.0}"
JUGGLE_RESTITUTION="${JUGGLE_RESTITUTION:-0.58}"
JUGGLE_TANGENTIAL_TRANSFER="${JUGGLE_TANGENTIAL_TRANSFER:-0.20}"
JUGGLE_DRAG_COEFFICIENT_PER_M="${JUGGLE_DRAG_COEFFICIENT_PER_M:-0.143}"
JUGGLE_BALL_RADIUS_M="${JUGGLE_BALL_RADIUS_M:-0.020}"
JUGGLE_READY_XY_TOLERANCE_M="${JUGGLE_READY_XY_TOLERANCE_M:-0.045}"
JUGGLE_READY_HEIGHT_TOLERANCE_M="${JUGGLE_READY_HEIGHT_TOLERANCE_M:-0.012}"
JUGGLE_READY_MAX_BALL_SPEED_M_S="${JUGGLE_READY_MAX_BALL_SPEED_M_S:-0.12}"
JUGGLE_READY_SETTLE_S="${JUGGLE_READY_SETTLE_S:-0.30}"
JUGGLE_MAX_BALL_AGE_S="${JUGGLE_MAX_BALL_AGE_S:-0.08}"
JUGGLE_CONTACT_GRACE_S="${JUGGLE_CONTACT_GRACE_S:-0.06}"
ENABLE_SAFETY_LIMITER="${ENABLE_SAFETY_LIMITER:-false}"
SAFETY_PARKING_VELOCITY_SCALE="${SAFETY_PARKING_VELOCITY_SCALE:-0.08}"
SAFETY_PARKING_ACCELERATION_SCALE="${SAFETY_PARKING_ACCELERATION_SCALE:-0.06}"
ENABLE_COMPENSATION="${ENABLE_COMPENSATION:-true}"
COMPENSATION_MODE="${COMPENSATION_MODE:-inverse_mpc}"
COMP_DELAY_MS="${COMP_DELAY_MS:-72.0}"
COMP_ACTUATOR_TAU_S="${COMP_ACTUATOR_TAU_S:-0.074}"
COMP_ACTUATOR_GAIN="${COMP_ACTUATOR_GAIN:-1.0}"
COMP_BUFFER_EXTRA_STEPS="${COMP_BUFFER_EXTRA_STEPS:-4}"
COMP_DELAY_CONDITIONING="${COMP_DELAY_CONDITIONING:-true}"
COMP_MPC_BETA="${COMP_MPC_BETA:-1.2}"
COMP_MPC_DELAY_SCALE="${COMP_MPC_DELAY_SCALE:-1.05}"
COMP_MPC_TAU_SCALE="${COMP_MPC_TAU_SCALE:-0.75}"
COMP_MPC_HORIZON_STEPS="${COMP_MPC_HORIZON_STEPS:-6}"
COMP_MPC_TRACKING_WEIGHT="${COMP_MPC_TRACKING_WEIGHT:-1.0}"
COMP_MPC_NOMINAL_WEIGHT="${COMP_MPC_NOMINAL_WEIGHT:-0.25}"
COMP_MPC_DELTA_WEIGHT="${COMP_MPC_DELTA_WEIGHT:-0.05}"
COMP_MPC_MAX_DELTA_RAD="${COMP_MPC_MAX_DELTA_RAD:-0.5235987755982988}"
if [ -z "${COMPENSATION_PRESET:-}" ]; then
    case "${POLICY_PROFILE,,}" in
        simcode2_idealpd67)
            COMPENSATION_PRESET="simcode2_live_v2"
            ;;
        simcode3|simcode3_actuator67|actuator67|actuator_67)
            COMPENSATION_PRESET="simcode3_checkpoint"
            ;;
        simcode4|simcode4_servo67|servo67|servo_67)
            COMPENSATION_PRESET="simcode4_checkpoint"
            ;;
        simcode5|simcode5_pure_actuator67|pure_actuator67|pure_actuator_67)
            COMPENSATION_PRESET="simcode5_direct"
            ;;
        *)
            COMPENSATION_PRESET="current"
            ;;
    esac
fi
case "$COMPENSATION_PRESET" in
    current)
        ;;
    simcode2_live_v2)
        # Live real-robot A/B tuning against the sourceSeq-aligned ideal-PD67
        # sim mirror.  The 6 s validation run retained 1114/1114 aligned
        # policy ticks with no done event; the observation-only 72 ms history
        # remains unchanged and this delay is used only by the compensator.
        COMPENSATION_MODE="inverse_mpc"
        COMP_DELAY_MS="72.0"
        COMP_ACTUATOR_TAU_S="0.074"
        COMP_ACTUATOR_GAIN="1.0"
        COMP_MPC_BETA="0.9"
        COMP_MPC_DELAY_SCALE="1.0"
        COMP_MPC_TAU_SCALE="0.75"
        COMP_MPC_HORIZON_STEPS="6"
        COMP_MPC_TRACKING_WEIGHT="1.0"
        COMP_MPC_NOMINAL_WEIGHT="0.30"
        COMP_MPC_DELTA_WEIGHT="0.10"
        COMP_MPC_MAX_DELTA_RAD="0.35"
        ;;
    simcode2_offline_v1)
        # Cross-run fit on probe_recording_20260722_153543/153937.  The fit
        # uses the common first 10 seconds and discards 20-29% of each tail so
        # differing terminal safety behaviour cannot bias the plant model.
        COMPENSATION_MODE="inverse_mpc"
        COMP_DELAY_MS="65.0"
        COMP_ACTUATOR_TAU_S="0.160"
        COMP_ACTUATOR_GAIN="1.0"
        COMP_MPC_BETA="0.5"
        COMP_MPC_DELAY_SCALE="1.0"
        COMP_MPC_TAU_SCALE="1.0"
        COMP_MPC_HORIZON_STEPS="8"
        COMP_MPC_TRACKING_WEIGHT="1.0"
        COMP_MPC_NOMINAL_WEIGHT="0.2"
        COMP_MPC_DELTA_WEIGHT="0.1"
        COMP_MPC_MAX_DELTA_RAD="0.25"
        ;;
    simcode3_checkpoint)
        # Checkpoint-authoritative real-robot contract.  The physical joints
        # are the fitted actuator plant: keep inverse-MPC and the 72 ms command
        # history, but never add the simulation's delay queue/FOPDT filter to
        # the command sent to the robot.
        COMPENSATION_MODE="inverse_mpc"
        CONTROL_RATE_HZ="200.0"
        POLICY_OBS_DIM="67"
        COMP_DELAY_MS="72.0"
        COMP_ACTUATOR_TAU_S="0.074"
        COMP_ACTUATOR_GAIN="1.0"
        COMP_BUFFER_EXTRA_STEPS="4"
        COMP_DELAY_CONDITIONING="true"
        COMP_MPC_BETA="1.2"
        COMP_MPC_DELAY_SCALE="1.05"
        COMP_MPC_TAU_SCALE="0.75"
        COMP_MPC_HORIZON_STEPS="6"
        COMP_MPC_TRACKING_WEIGHT="1.0"
        COMP_MPC_NOMINAL_WEIGHT="0.25"
        COMP_MPC_DELTA_WEIGHT="0.05"
        COMP_MPC_MAX_DELTA_RAD="0.5235987755982988"
        # Real-only balanced candidate from the sourceSeq-aligned 1200-step
        # drag-ball trial on 2026-07-29 (exact common start, active-reset arm):
        # joint-position/action/command-position gaps improved by
        # 10.3%/4.1%/13.3% versus the checkpoint defaults while joint velocity
        # and acceleration gaps remained within 0.4%.  The sim contract above
        # remains the checkpoint values and is routed independently.
        REAL_COMP_DELAY_MS="${REAL_COMP_DELAY_MS:-67.75}"
        REAL_COMP_ACTUATOR_TAU_S="${REAL_COMP_ACTUATOR_TAU_S:-0.095}"
        REAL_COMP_ACTUATOR_GAIN="${REAL_COMP_ACTUATOR_GAIN:-1.01075}"
        REAL_COMP_MPC_BETA="${REAL_COMP_MPC_BETA:-1.19375}"
        REAL_COMP_MPC_DELAY_SCALE="${REAL_COMP_MPC_DELAY_SCALE:-1.0815}"
        REAL_COMP_MPC_TAU_SCALE="${REAL_COMP_MPC_TAU_SCALE:-0.776}"
        REAL_COMP_MPC_HORIZON_STEPS="${REAL_COMP_MPC_HORIZON_STEPS:-6}"
        REAL_COMP_MPC_TRACKING_WEIGHT="${REAL_COMP_MPC_TRACKING_WEIGHT:-1.0}"
        REAL_COMP_MPC_NOMINAL_WEIGHT="${REAL_COMP_MPC_NOMINAL_WEIGHT:-0.22925}"
        REAL_COMP_MPC_DELTA_WEIGHT="${REAL_COMP_MPC_DELTA_WEIGHT:-0.0575}"
        REAL_COMP_MPC_MAX_DELTA_RAD="${REAL_COMP_MPC_MAX_DELTA_RAD:-0.46995}"
        ;;
    simcode4_checkpoint)
        # Frozen simcode4 contract. Unlike the locally tuned simcode3 real
        # preset, keep the checkpoint inverse-MPC parameters unchanged; the
        # frozen MJX uses a downstream target-aware servo planner at velocity
        # scale 1.0 and acceleration scale 0.8.  Hardware defaults to sending
        # the compensated angle directly; neither planner nor FOPDT is replayed
        # in software. A compatible V6 may explicitly opt into the planner.
        COMPENSATION_MODE="inverse_mpc"
        CONTROL_RATE_HZ="200.0"
        POLICY_OBS_DIM="67"
        COMP_DELAY_MS="72.0"
        COMP_ACTUATOR_TAU_S="0.074"
        COMP_ACTUATOR_GAIN="1.0"
        COMP_BUFFER_EXTRA_STEPS="4"
        COMP_DELAY_CONDITIONING="true"
        COMP_MPC_BETA="1.2"
        COMP_MPC_DELAY_SCALE="1.05"
        COMP_MPC_TAU_SCALE="0.75"
        COMP_MPC_HORIZON_STEPS="6"
        COMP_MPC_TRACKING_WEIGHT="1.0"
        COMP_MPC_NOMINAL_WEIGHT="0.25"
        COMP_MPC_DELTA_WEIGHT="0.05"
        COMP_MPC_MAX_DELTA_RAD="0.5235987755982988"
        ;;
    simcode5_direct)
        # The physical joints are the actuator plant represented by the
        # checkpoint's per-joint delay + second-order simulation model.
        # Keep delayed history for observation only and send the latest angle.
        ENABLE_COMPENSATION="false"
        COMPENSATION_MODE="none"
        CONTROL_RATE_HZ="200.0"
        POLICY_OBS_DIM="67"
        COMP_DELAY_MS="45.0"
        COMP_DELAY_CONDITIONING="true"
        ;;
    *)
        echo "Unsupported COMPENSATION_PRESET=$COMPENSATION_PRESET (expected current, simcode2_live_v2, simcode2_offline_v1, simcode3_checkpoint, simcode4_checkpoint, or simcode5_direct)" >&2
        exit 2
        ;;
esac
REAL_COMP_DELAY_MS="${REAL_COMP_DELAY_MS:-$COMP_DELAY_MS}"
REAL_COMP_ACTUATOR_TAU_S="${REAL_COMP_ACTUATOR_TAU_S:-$COMP_ACTUATOR_TAU_S}"
REAL_COMP_ACTUATOR_GAIN="${REAL_COMP_ACTUATOR_GAIN:-$COMP_ACTUATOR_GAIN}"
REAL_COMP_MPC_BETA="${REAL_COMP_MPC_BETA:-$COMP_MPC_BETA}"
REAL_COMP_MPC_DELAY_SCALE="${REAL_COMP_MPC_DELAY_SCALE:-$COMP_MPC_DELAY_SCALE}"
REAL_COMP_MPC_TAU_SCALE="${REAL_COMP_MPC_TAU_SCALE:-$COMP_MPC_TAU_SCALE}"
REAL_COMP_MPC_HORIZON_STEPS="${REAL_COMP_MPC_HORIZON_STEPS:-$COMP_MPC_HORIZON_STEPS}"
REAL_COMP_MPC_TRACKING_WEIGHT="${REAL_COMP_MPC_TRACKING_WEIGHT:-$COMP_MPC_TRACKING_WEIGHT}"
REAL_COMP_MPC_NOMINAL_WEIGHT="${REAL_COMP_MPC_NOMINAL_WEIGHT:-$COMP_MPC_NOMINAL_WEIGHT}"
REAL_COMP_MPC_DELTA_WEIGHT="${REAL_COMP_MPC_DELTA_WEIGHT:-$COMP_MPC_DELTA_WEIGHT}"
REAL_COMP_MPC_MAX_DELTA_RAD="${REAL_COMP_MPC_MAX_DELTA_RAD:-$COMP_MPC_MAX_DELTA_RAD}"
if [ -z "${BALL_OBSERVATION_RATE_HZ:-}" ]; then
    # One experiment-wide camera/simulation observation contract.  The
    # controller still runs at 200 Hz and holds each 90 Hz ball sample between
    # refreshes.
    BALL_OBSERVATION_RATE_HZ="90.0"
fi
PROBE_BACKEND_CONTROL_LOOP="${PROBE_BACKEND_CONTROL_LOOP:-true}"
PROBE_CONTROL_SUBPROCESS="${PROBE_CONTROL_SUBPROCESS:-$PROBE_BACKEND_CONTROL_LOOP}"
PROBE_CONTROL_SUBPROCESS_STARTUP_TIMEOUT_S="${PROBE_CONTROL_SUBPROCESS_STARTUP_TIMEOUT_S:-300.0}"
PROBE_BACKEND_SNAPSHOT_RATE_HZ="${PROBE_BACKEND_SNAPSHOT_RATE_HZ:-60.0}"
PROBE_STEP_RATE_HZ="${PROBE_STEP_RATE_HZ:-60.0}"
PROBE_DISPLAY_RATE_HZ="${PROBE_DISPLAY_RATE_HZ:-60.0}"
PROBE_RECORDING_OUTPUT_ROOT="${PROBE_RECORDING_OUTPUT_ROOT:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings}"
ANALYSIS="${ANALYSIS:-false}"
ANALYSIS_RUN_DIR="${ANALYSIS_RUN_DIR:-}"
PROBE_RECORDING_CONTROL_RATE_HZ="${PROBE_RECORDING_CONTROL_RATE_HZ:-200.0}"
PROBE_RECORDING_IMAGE_RATE_HZ="${PROBE_RECORDING_IMAGE_RATE_HZ:-60.0}"
PROBE_RECORDING_ANALYSIS_VIDEO_FPS="${PROBE_RECORDING_ANALYSIS_VIDEO_FPS:-60.0}"
PROBE_RECORDING_QUEUE_SIZE="${PROBE_RECORDING_QUEUE_SIZE:-4096}"
ENABLE_SIM_MIRROR="${ENABLE_SIM_MIRROR:-true}"
SIM_MIRROR_DEFAULT_ENABLED="${SIM_MIRROR_DEFAULT_ENABLED:-$ENABLE_SIM_MIRROR}"
SIM_SUBPROCESS="${SIM_SUBPROCESS:-true}"
# The legacy/simcode3/simcode5 profiles use the lightweight MuJoCo mirror by
# default.  The simcode5 CPU mirror retains its per-joint delay and second-order
# actuator plant.  Its original MJX env remains available with SIM_BACKEND=mjx.
# The simcode4 frozen contract is enforced below to use its original MJX env,
# including the target-aware planner inside the simulated plant.
SIM_BACKEND="${SIM_BACKEND:-mujoco}"
case "${POLICY_PROFILE,,}" in
    simcode4|simcode4_servo67|servo67|servo_67)
        SIM_BACKEND="mjx"
        ;;
esac
SIM_XML_PATH="${SIM_XML_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
SIMCODE2_SIM_XML_PATH="${SIMCODE2_SIM_XML_PATH:-$WORKSPACE_DIR/simcode2/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
SIMCODE3_SIM_XML_PATH="${SIMCODE3_SIM_XML_PATH:-$WORKSPACE_DIR/simcode3/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
SIMCODE4_SIM_XML_PATH="${SIMCODE4_SIM_XML_PATH:-$WORKSPACE_DIR/simcode4/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
SIMCODE5_SIM_XML_PATH="${SIMCODE5_SIM_XML_PATH:-$WORKSPACE_DIR/simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml}"
SIM_ENABLE_VIDEO="${SIM_ENABLE_VIDEO:-false}"
SIM_FRAME_RATE_HZ="${SIM_FRAME_RATE_HZ:-90.0}"
SIM_RENDER_WIDTH="${SIM_RENDER_WIDTH:-640}"
SIM_RENDER_HEIGHT="${SIM_RENDER_HEIGHT:-360}"
SIM_USE_MUJOCO_RENDERER="${SIM_USE_MUJOCO_RENDERER:-false}"
SIM_CATCHUP_BATCH_STEPS="${SIM_CATCHUP_BATCH_STEPS:-1}"
SIM_COMMAND_QUEUE_SIZE="${SIM_COMMAND_QUEUE_SIZE:-256}"
SIM_STARTUP_TIMEOUT_S="${SIM_STARTUP_TIMEOUT_S:-$PROBE_CONTROL_SUBPROCESS_STARTUP_TIMEOUT_S}"
SIM_RECORDING_FLUSH_TIMEOUT_S="${SIM_RECORDING_FLUSH_TIMEOUT_S:-5.0}"
# Policy-facing 3D Kalman filter. Prediction can be disabled while keeping
# measurement fusion/denoising enabled; disabling the estimator bypasses both.
ENABLE_BALL_STATE_ESTIMATOR="${ENABLE_BALL_STATE_ESTIMATOR:-false}"
# Opt-in backend selector.  Keeping this false preserves the legacy estimator
# whenever the estimator master switch is enabled.
ENABLE_HYBRID_BALL_TRACKER="${ENABLE_HYBRID_BALL_TRACKER:-false}"
HYBRID_BALL_TRACKER_CONFIG_PATH="${HYBRID_BALL_TRACKER_CONFIG_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/ball_tracker/config.yaml}"
BALL_ESTIMATOR_ENABLE_PREDICTION="${BALL_ESTIMATOR_ENABLE_PREDICTION:-false}"
BALL_ESTIMATOR_ENABLE_AERO_DRAG="${BALL_ESTIMATOR_ENABLE_AERO_DRAG:-true}"
BALL_ESTIMATOR_MAX_EXTRAPOLATION_S="${BALL_ESTIMATOR_MAX_EXTRAPOLATION_S:-0.45}"
BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S="${BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S:-0.08}"
# continuous fills every control tick with a fresh prediction;
# predict_on_frame_hold refreshes a predicted snapshot at the configured rate
# and zero-order holds it between policy ticks.
BALL_ESTIMATOR_CONTROL_OUTPUT_MODE="${BALL_ESTIMATOR_CONTROL_OUTPUT_MODE:-continuous}"
BALL_ESTIMATOR_FRAME_HOLD_OUTPUT_RATE_HZ="${BALL_ESTIMATOR_FRAME_HOLD_OUTPUT_RATE_HZ:-50.0}"
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS="${BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS:-true}"
RL_USE_VALID_BALL_OBS_AGE="${RL_USE_VALID_BALL_OBS_AGE:-true}"
case "${POLICY_PROFILE,,}" in
simcode3|simcode3_actuator67|actuator67|actuator_67)
    # obs[49] and the causal contact phase require age since the last valid
    # 60 Hz ball frame; this is part of the checkpoint input contract.
    RL_USE_VALID_BALL_OBS_AGE="true"
    ;;
simcode4|simcode4_servo67|servo67|servo_67)
    # The checkpoint never ages a held non-missing 60 Hz observation.
    RL_USE_VALID_BALL_OBS_AGE="false"
    ;;
simcode5|simcode5_pure_actuator67|pure_actuator67|pure_actuator_67)
    # The checkpoint never ages a held non-missing 60 Hz observation.
    RL_USE_VALID_BALL_OBS_AGE="false"
    ;;
esac
BALL_RELEASE_USE_ESTIMATOR_STATE="${BALL_RELEASE_USE_ESTIMATOR_STATE:-false}"
BALL_ESTIMATOR_TRUST_LOSS_S="${BALL_ESTIMATOR_TRUST_LOSS_S:-0.10}"
BALL_ESTIMATOR_DEGRADE_LOSS_S="${BALL_ESTIMATOR_DEGRADE_LOSS_S:-0.40}"
BALL_ESTIMATOR_INVALID_AFTER_S="${BALL_ESTIMATOR_INVALID_AFTER_S:-0.60}"
BALL_ESTIMATOR_PREDICTION_HORIZON_S="${BALL_ESTIMATOR_PREDICTION_HORIZON_S:-0.0}"
BALL_ESTIMATOR_Q_POS_MM2_S="${BALL_ESTIMATOR_Q_POS_MM2_S:-500.0}"
BALL_ESTIMATOR_Q_VEL_MULT="${BALL_ESTIMATOR_Q_VEL_MULT:-100.0}"
BALL_ESTIMATOR_Q_DEPTH_POS_MULT="${BALL_ESTIMATOR_Q_DEPTH_POS_MULT:-0.50}"
BALL_ESTIMATOR_Q_DEPTH_VEL_MULT="${BALL_ESTIMATOR_Q_DEPTH_VEL_MULT:-0.20}"
BALL_ESTIMATOR_R_LATERAL_MM="${BALL_ESTIMATOR_R_LATERAL_MM:-25.0}"
BALL_ESTIMATOR_R_DEPTH_MM="${BALL_ESTIMATOR_R_DEPTH_MM:-90.0}"
ENABLE_ROS_BRIDGE="${ENABLE_ROS_BRIDGE:-true}"
ROS_PUBLISH_COMMANDS="${ROS_PUBLISH_COMMANDS:-true}"
COMMAND_TOPIC="${COMMAND_TOPIC:-/mx_mix_command}"
JOINT_STATE_TOPIC="${JOINT_STATE_TOPIC:-/joint_states}"
CART_STATE_TOPIC="${CART_STATE_TOPIC:-/cart_states}"
RIGHT_ARM_MU_IDX="${RIGHT_ARM_MU_IDX:-3}"
ENABLE_CAMERA_BALL="${ENABLE_CAMERA_BALL:-true}"
ENABLE_MOCAP_BALL="${ENABLE_MOCAP_BALL:-true}"
MOCAP_UDP_HOST="${MOCAP_UDP_HOST:-0.0.0.0}"
MOCAP_UDP_PORT="${MOCAP_UDP_PORT:-9870}"
MOCAP_CALIBRATION_PATH="${MOCAP_CALIBRATION_PATH:-$WORKSPACE_DIR/mocap_calibration.txt}"
MOCAP_MARKER_ID="${MOCAP_MARKER_ID:-2}"
MOCAP_RIGID_BODY_ID="${MOCAP_RIGID_BODY_ID:-1}"
MOCAP_RIGID_BODY_SLOT="${MOCAP_RIGID_BODY_SLOT:-$MOCAP_RIGID_BODY_ID}"
MOCAP_RIGID_BODY_NAME="${MOCAP_RIGID_BODY_NAME:-}"
MOCAP_POSITION_UNIT="${MOCAP_POSITION_UNIT:-auto}"
MOCAP_TIMESTAMP_UNIT="${MOCAP_TIMESTAMP_UNIT:-s}"
MOCAP_TIMEOUT_S="${MOCAP_TIMEOUT_S:-0.5}"
MOCAP_VELOCITY_ALPHA="${MOCAP_VELOCITY_ALPHA:-0.35}"
MOCAP_MAX_VELOCITY_M_S="${MOCAP_MAX_VELOCITY_M_S:-20.0}"
CAMERA_SOURCE_MODE="${CAMERA_SOURCE_MODE:-direct_realsense}"
USE_HYBRID_DETECTOR="${USE_HYBRID_DETECTOR:-true}"
USE_DEPTH_SENSOR="${USE_DEPTH_SENSOR:-true}"
IMAGE_WIDTH="${IMAGE_WIDTH:-848}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
REALSENSE_FPS="${REALSENSE_FPS:-90}"
CAMERA_IMAGE_TOPIC="${CAMERA_IMAGE_TOPIC:-/camera/camera/color/image_raw/compressed}"
CAMERA_RAW_IMAGE_TOPIC="${CAMERA_RAW_IMAGE_TOPIC:-/camera/camera/color/image_raw}"
CAMERA_IMAGE_TRANSPORT="${CAMERA_IMAGE_TRANSPORT:-auto}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/camera/color/camera_info}"
CALIBRATION_JSON_PATH="${CALIBRATION_JSON_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/vision_calib/eye_hand_result_848_undistorted.json}"
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/models/best.pt}"
TARGET_CLASS="${TARGET_CLASS:-Class 0}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
IR_PARAMS_JSON="${IR_PARAMS_JSON:-$WORKSPACE_DIR/ir_params.json}"
ENABLE_YOLO="${ENABLE_YOLO:-false}"
YOLO_INTERVAL="${YOLO_INTERVAL:-1}"
# Keep the default IR path consistent with ir_tuning_tool.py.  The optional
# candidate filter adds absolute brightness/depth hard gates that are not part
# of the tuning tool; in particular, a valid distant/small ball can have a
# contour brightness well below the historical 23000 threshold.
ENABLE_HYBRID_CANDIDATE_FILTER="${ENABLE_HYBRID_CANDIDATE_FILTER:-false}"
CANDIDATE_MIN_BRIGHTNESS="${CANDIDATE_MIN_BRIGHTNESS:-23000.0}"
CANDIDATE_MAX_DEPTH_DISAGREEMENT_MM="${CANDIDATE_MAX_DEPTH_DISAGREEMENT_MM:-180.0}"
CANDIDATE_MOTION_SCALE_PX="${CANDIDATE_MOTION_SCALE_PX:-130.0}"
IR_THRESHOLD="${IR_THRESHOLD:-170}"
USE_ADAPTIVE_IR="${USE_ADAPTIVE_IR:-true}"
IR_EXPOSURE="${IR_EXPOSURE:-1000}"
IR_GAIN="${IR_GAIN:-16}"
IR_LASER_POWER="${IR_LASER_POWER:-360}"
IR_MIN_RADIUS="${IR_MIN_RADIUS:-15}"
IR_MAX_RADIUS="${IR_MAX_RADIUS:-60}"
IR_CIRCULARITY_MIN="${IR_CIRCULARITY_MIN:-0.7}"
IR_BRIGHTNESS_PERCENTILE="${IR_BRIGHTNESS_PERCENTILE:-25}"
HYBRID_ROI_MARGIN="${HYBRID_ROI_MARGIN:-100}"
DEPTH_CORRECTION_MODEL_PATH="${DEPTH_CORRECTION_MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/depth_correction_model.pkl}"
DEPTH_NEIGHBORHOOD_PX="${DEPTH_NEIGHBORHOOD_PX:-2}"
DEPTH_TEMPORAL_WINDOW_FRAMES="${DEPTH_TEMPORAL_WINDOW_FRAMES:-3}"
INFER_IMGSZ="${INFER_IMGSZ:-640}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-0}"
DETECTOR_HALF="${DETECTOR_HALF:-false}"
# Camera filter defaults. All remain independently selectable in the web UI;
# these values only set their initial states.
USE_KALMAN="${USE_KALMAN:-false}"
USE_IR_DEPTH_KALMAN="${USE_IR_DEPTH_KALMAN:-false}"
USE_SYNCHRONIZED_IR_DEPTH="${USE_SYNCHRONIZED_IR_DEPTH:-false}"
CAMERA_TILT_DEG="${CAMERA_TILT_DEG:-50.33}"
USE_UNDISTORT="${USE_UNDISTORT:-true}"
ENABLE_ELLIPSE_FIT="${ENABLE_ELLIPSE_FIT:-true}"
ENABLE_TEMPORAL_TRACKER="${ENABLE_TEMPORAL_TRACKER:-false}"
TRACKER_MAX_MISS_BRIDGE="${TRACKER_MAX_MISS_BRIDGE:-2}"
TRACKER_GATE_PX="${TRACKER_GATE_PX:-120.0}"
HSV_LOWER="${HSV_LOWER:-5,50,50}"
HSV_UPPER="${HSV_UPPER:-30,255,255}"
COLOR_MATCH_THRESHOLD="${COLOR_MATCH_THRESHOLD:-0.0}"
COLOR_CIRCULARITY_MIN="${COLOR_CIRCULARITY_MIN:-0.0}"
ENABLE_CAMERA_DETECTION="${ENABLE_CAMERA_DETECTION:-true}"
ENABLE_CAMERA_VISUALIZATION="${ENABLE_CAMERA_VISUALIZATION:-true}"
CAMERA_VISUALIZATION_RATE_HZ="${CAMERA_VISUALIZATION_RATE_HZ:-30.0}"
CAMERA_WEB_MAX_WIDTH="${CAMERA_WEB_MAX_WIDTH:-848}"
CAMERA_WEB_JPEG_QUALITY="${CAMERA_WEB_JPEG_QUALITY:-62}"
ENABLE_CAMERA_COLOR_FILTER="${ENABLE_CAMERA_COLOR_FILTER:-false}"
SIM_CARTESIAN_X_OFFSET_M="${SIM_CARTESIAN_X_OFFSET_M:-0.0}"
SIM_CARTESIAN_Y_OFFSET_M="${SIM_CARTESIAN_Y_OFFSET_M:-0.023583}"
SIM_BASE_HEIGHT_M="${SIM_BASE_HEIGHT_M:-1.302396}"

if [ "$ENABLE_HYBRID_BALL_TRACKER" = "true" ]; then
    if [ "$ENABLE_BALL_STATE_ESTIMATOR" != "true" ]; then
        echo "ERROR: ENABLE_HYBRID_BALL_TRACKER=true requires ENABLE_BALL_STATE_ESTIMATOR=true" >&2
        echo "       Example: ENABLE_BALL_STATE_ESTIMATOR=true ENABLE_HYBRID_BALL_TRACKER=true BALL_ESTIMATOR_ENABLE_PREDICTION=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh" >&2
        exit 2
    fi
    if [ ! -f "$HYBRID_BALL_TRACKER_CONFIG_PATH" ]; then
        echo "ERROR: hybrid ball tracker config not found: $HYBRID_BALL_TRACKER_CONFIG_PATH" >&2
        exit 2
    fi
    BALL_ESTIMATOR_BACKEND="hybrid_ekf9"
else
    BALL_ESTIMATOR_BACKEND="legacy_kf6"
fi

if [ "$LOAD_CONFIG_DONE_MONITORING" = "true" ]; then
    echo "[ProbeConfig] loaded $CONFIG_DONE_MONITORING_YAML (${#_YAML_EFFECTIVE_KEYS[@]} YAML values applied; explicit environment wins)"
else
    echo "[ProbeConfig] YAML loading disabled; using environment/script defaults"
fi
echo "[ProbeConfig] workspace mode=$WORKSPACE_SAFETY_MODE ee_x=$DONE_EE_X_LIMITS_M ee_y=$DONE_EE_Y_LIMITS_M ee_z=$DONE_EE_Z_LIMITS_M"
case "$WORKSPACE_SAFETY_MODE" in
    *_soft)
        echo "[ProbeConfig] workspace safety=hard_envelope terminal_action=reset_joint_pose_on_any_done pose_source=joint_state_fk reset_initial_pose=$RESET_INITIAL_POSE"
        ;;
    position|predictive)
        echo "[ProbeConfig] workspace terminal action=measured_hold_then_stop"
        ;;
esac
echo "[ProbeConfig] predictive=$DONE_ENABLE_PREDICTIVE_STOP horizon=${DONE_PREDICTION_HORIZON_S}s dt=${DONE_PREDICTION_DT_S}s margin=${DONE_PREDICTION_MARGIN_M}m ori_margin=${DONE_PREDICTION_ORIENTATION_MARGIN_RAD}rad done_monitoring=$ENABLE_DONE_MONITORING"
echo "[ProbeConfig] ball estimator=$ENABLE_BALL_STATE_ESTIMATOR backend=$BALL_ESTIMATOR_BACKEND prediction=$BALL_ESTIMATOR_ENABLE_PREDICTION output=$BALL_ESTIMATOR_CONTROL_OUTPUT_MODE"
if [ "$ENABLE_HYBRID_BALL_TRACKER" = "true" ]; then
    echo "[ProbeConfig] hybrid tracker config=$HYBRID_BALL_TRACKER_CONFIG_PATH"
fi
echo "[ProbeConfig] reset_initial_pose=$RESET_INITIAL_POSE"
echo "[ProbeConfig] RL profile=$POLICY_PROFILE legacy_model=$MODEL_PATH"
echo "[ProbeConfig] simcode2 ideal-PD67 model=$SIMCODE2_MODEL_PATH"
echo "[ProbeConfig] simcode3 actuator67 model=$SIMCODE3_MODEL_PATH"
echo "[ProbeConfig] simcode4 servo67 model=$SIMCODE4_MODEL_PATH frozen_sim_order=FOPDT->planner real_servo_planner=$SIMCODE4_REAL_SERVO_PLANNER (default off)"
echo "[ProbeConfig] simcode4 real default order=compensated_angle->physical_joint; opt-in planner requires checkpoint env_cfg planner_before_actuator=true"
echo "[ProbeConfig] simcode5 pure-actuator67 model=$SIMCODE5_MODEL_PATH real_order=latest_policy_angle->physical_joint mirror_plant=per_joint_delay->second_order->sport_PD"
echo "[ProbeConfig] sim compensation_preset=$COMPENSATION_PRESET mode=$COMPENSATION_MODE delay=${COMP_DELAY_MS}ms tau=${COMP_ACTUATOR_TAU_S}s gain=$COMP_ACTUATOR_GAIN beta=$COMP_MPC_BETA tau_scale=$COMP_MPC_TAU_SCALE horizon=$COMP_MPC_HORIZON_STEPS weights=${COMP_MPC_TRACKING_WEIGHT}/${COMP_MPC_NOMINAL_WEIGHT}/${COMP_MPC_DELTA_WEIGHT} max_delta=${COMP_MPC_MAX_DELTA_RAD}rad"
echo "[ProbeConfig] real-only compensation delay=${REAL_COMP_DELAY_MS}ms tau=${REAL_COMP_ACTUATOR_TAU_S}s gain=$REAL_COMP_ACTUATOR_GAIN beta=$REAL_COMP_MPC_BETA delay_scale=$REAL_COMP_MPC_DELAY_SCALE tau_scale=$REAL_COMP_MPC_TAU_SCALE horizon=$REAL_COMP_MPC_HORIZON_STEPS weights=${REAL_COMP_MPC_TRACKING_WEIGHT}/${REAL_COMP_MPC_NOMINAL_WEIGHT}/${REAL_COMP_MPC_DELTA_WEIGHT} max_delta=$REAL_COMP_MPC_MAX_DELTA_RAD parameter_target=real"
echo "[ProbeConfig] policy observation=${POLICY_OBS_DIM}D ball_refresh=${BALL_OBSERVATION_RATE_HZ}Hz valid_frame_age=$RL_USE_VALID_BALL_OBS_AGE joint_feedback=measured_q_dq"
echo "[ProbeConfig] camera=${CAMERA_SOURCE_MODE} ${IMAGE_WIDTH}x${IMAGE_HEIGHT}@${REALSENSE_FPS}Hz yolo=${ENABLE_YOLO} candidate_filter=${ENABLE_HYBRID_CANDIDATE_FILTER} visualization=${ENABLE_CAMERA_VISUALIZATION}@${CAMERA_VISUALIZATION_RATE_HZ}Hz"
echo "[ProbeConfig] mocap=${ENABLE_MOCAP_BALL} udp=${MOCAP_UDP_HOST}:${MOCAP_UDP_PORT} ball_marker=${MOCAP_MARKER_ID} rb_slot=${MOCAP_RIGID_BODY_SLOT} rb_id=${MOCAP_RIGID_BODY_ID} calibration=${MOCAP_CALIBRATION_PATH}"
echo "[ProbeConfig] controller=$DEFAULT_CONTROL_MODE tracking_gap=${RACKET_TRACKING_HEIGHT_OFFSET_M}m tracking_mpc=$RACKET_TRACKING_USE_MPC tracking_ball_age=${RACKET_TRACKING_MAX_BALL_AGE_S}s"
echo "[ProbeConfig] shared_racket_mpc=N${RACKET_TRACKING_MPC_HORIZON_STEPS} vmax=${RACKET_TRACKING_MAX_CARTESIAN_SPEED_M_S}m/s amax=${RACKET_TRACKING_MAX_CARTESIAN_ACCELERATION_M_S2}m/s2 joint_scale=${RACKET_TRACKING_JOINT_VELOCITY_SCALE}/${RACKET_TRACKING_JOINT_ACCELERATION_SCALE}"
echo "[ProbeConfig] mpc_juggle=start_off countdown=${JUGGLE_COUNTDOWN_S}s seed_apex=${JUGGLE_SEED_APEX_HEIGHT_M}m target_apex=${JUGGLE_TARGET_APEX_HEIGHT_M}m restitution=${JUGGLE_RESTITUTION} contact_grace=${JUGGLE_CONTACT_GRACE_S}s"
if [ "${PROBE_CONFIG_ONLY:-false}" = "true" ]; then
    exit 0
fi

ROS_BRIDGE_ARGS=()
CAMERA_ARGS=()
MOCAP_ARGS=()
FLOW_ARGS=()
export PROBE_CONTROL_SUBPROCESS_STARTUP_TIMEOUT_S
if [ "$ENABLE_SIM_MIRROR" = "prompt" ] && [ -t 0 ]; then
    read -r -p "Start sim mirror for sim2real recording? [y/N] " _sim_answer
    case "$_sim_answer" in
        y|Y|yes|YES) ENABLE_SIM_MIRROR="true" ;;
        *) ENABLE_SIM_MIRROR="false" ;;
    esac
    if [ "$ENABLE_SIM_MIRROR" = "true" ]; then
        _sim_backend_default="$SIM_BACKEND"
        case "$_sim_backend_default" in
            mjx|MJX) _sim_backend_default="mjx" ;;
            mujoco|MUJOCO) _sim_backend_default="mujoco" ;;
            auto|AUTO) _sim_backend_default="auto" ;;
            *) _sim_backend_default="mjx" ;;
        esac
        read -r -p "Select sim backend [mjx/mujoco/auto] (default: $_sim_backend_default): " _sim_backend_answer
        _sim_backend_answer="${_sim_backend_answer:-$_sim_backend_default}"
        case "$_sim_backend_answer" in
            mjx|MJX) SIM_BACKEND="mjx" ;;
            mujoco|MUJOCO) SIM_BACKEND="mujoco" ;;
            auto|AUTO) SIM_BACKEND="auto" ;;
            *)
                echo "WARNING: unknown sim backend '$_sim_backend_answer'; using $_sim_backend_default" >&2
                SIM_BACKEND="$_sim_backend_default"
                ;;
        esac
    fi
    SIM_MIRROR_DEFAULT_ENABLED="$ENABLE_SIM_MIRROR"
fi

# A handwritten CPU mirror is not authoritative for simcode4.  Force the
# original MJX train/validate environment and do this before CUDA/JAX setup so
# an interactive choice of "mujoco" cannot accidentally omit its runtime env.
if [ "$ENABLE_SIM_MIRROR" = "true" ]; then
    case "${POLICY_PROFILE,,}" in
        simcode4|simcode4_servo67|servo67|servo_67)
            if [ "$SIM_BACKEND" != "mjx" ]; then
                echo "[ProbeConfig] simcode4 requires its frozen MJX mirror; overriding sim backend '$SIM_BACKEND' -> 'mjx'"
            fi
            SIM_BACKEND="mjx"
            ;;
    esac
fi
export SIM_CATCHUP_BATCH_STEPS
export SIM_COMMAND_QUEUE_SIZE
export SIM_STARTUP_TIMEOUT_S
export SIM_RECORDING_FLUSH_TIMEOUT_S

# pip-installed CUDA libraries (torch/JAX) live below site-packages/nvidia/*/lib
# and are not always present in the system loader path.  Camera detection needs
# cuDNN even when visualization is disabled, so configure this independently
# from the selected simulation backend.
if { [ "$ENABLE_CAMERA_BALL" = "true" ] || \
     { [ "$ENABLE_SIM_MIRROR" = "true" ] && [ "$SIM_BACKEND" = "mjx" ]; }; }; then
    _PY_SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
    if [ -n "$_PY_SITE_PACKAGES" ] && [ -d "$_PY_SITE_PACKAGES/nvidia" ]; then
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
fi

if [ "$ENABLE_SIM_MIRROR" = "true" ] && [ "$SIM_BACKEND" = "mjx" ]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
    export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/jax_compilation_cache}"
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
    export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:-0}"
    mkdir -p "$JAX_COMPILATION_CACHE_DIR"
fi
if [ "$ENABLE_BALL_CAPTURE_GATE" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-capture-gate)
fi
FLOW_ARGS+=(--workspace-safety-mode "$WORKSPACE_SAFETY_MODE")
if [ "$ENABLE_BALL_RELEASE_GATE" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-release-gate)
fi
if [ "$ENABLE_BALL_LOST_FAILSAFE" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-lost-failsafe)
fi
if [ "$ENABLE_DONE_MONITORING" != "true" ]; then
    FLOW_ARGS+=(--disable-done-monitoring)
fi
if [ "$DONE_ENABLE_PREDICTIVE_STOP" != "true" ]; then
    FLOW_ARGS+=(--disable-done-predictive-stop)
fi
if [ "$ENABLE_SAFETY_LIMITER" = "true" ]; then
    FLOW_ARGS+=(--enable-safety-limiter)
fi
if [ "$ENABLE_COMPENSATION" != "true" ]; then
    FLOW_ARGS+=(--disable-compensation)
fi
if [ "$SIMCODE4_REAL_SERVO_PLANNER" = "true" ]; then
    FLOW_ARGS+=(--simcode4-real-servo-planner)
fi
if [ "$RACKET_TRACKING_USE_MPC" = "true" ]; then
    FLOW_ARGS+=(--racket-tracking-use-mpc)
fi
if [ "$ENABLE_SIM_MIRROR" = "true" ]; then
    FLOW_ARGS+=(--enable-sim-mirror)
    if [ "$SIM_MIRROR_DEFAULT_ENABLED" != "true" ]; then
        FLOW_ARGS+=(--sim-mirror-default-disabled)
    fi
fi
if [ "$SIM_SUBPROCESS" != "true" ]; then
    FLOW_ARGS+=(--disable-sim-subprocess)
fi
if [ "$SIM_ENABLE_VIDEO" = "true" ]; then
    FLOW_ARGS+=(--enable-sim-video)
fi
if [ "$SIM_USE_MUJOCO_RENDERER" != "true" ]; then
    FLOW_ARGS+=(--disable-sim-mujoco-renderer)
fi
if [ "$COMP_DELAY_CONDITIONING" != "true" ]; then
    FLOW_ARGS+=(--disable-comp-delay-conditioning)
fi
if [ "$ENABLE_BALL_STATE_ESTIMATOR" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-state-estimator)
fi
if [ "$ENABLE_HYBRID_BALL_TRACKER" = "true" ]; then
    FLOW_ARGS+=(--enable-hybrid-ball-tracker)
fi
if [ "$BALL_ESTIMATOR_ENABLE_PREDICTION" = "true" ]; then
    FLOW_ARGS+=(--enable-ball-estimator-prediction)
else
    FLOW_ARGS+=(--disable-ball-estimator-prediction)
fi
if [ "$BALL_ESTIMATOR_ENABLE_AERO_DRAG" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-estimator-aero-drag)
fi
if [ "$RL_USE_VALID_BALL_OBS_AGE" != "true" ]; then
    FLOW_ARGS+=(--disable-rl-valid-ball-obs-age)
fi
if [ "$BALL_RELEASE_USE_ESTIMATOR_STATE" = "true" ]; then
    FLOW_ARGS+=(--ball-release-use-estimator-state)
fi
if [ "$BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS" = "true" ]; then
    FLOW_ARGS+=(--ball-estimator-release-gated-dynamics)
fi
if [ "$PROBE_BACKEND_CONTROL_LOOP" = "true" ]; then
    FLOW_ARGS+=(--backend-control-loop)
fi
if [ "$PROBE_CONTROL_SUBPROCESS" = "true" ]; then
    FLOW_ARGS+=(--control-subprocess --backend-control-loop)
fi
if [ "$ENABLE_ROS_BRIDGE" = "true" ] || [ "$ENABLE_CAMERA_BALL" = "true" ] || [ "$ENABLE_MOCAP_BALL" = "true" ]; then
    export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ros_log}"
    mkdir -p "$ROS_LOG_DIR"
    if [ -f "$SETUP_FILE" ]; then
        set +u
        source "$SETUP_FILE"
        set -u
    else
        echo "WARNING: ROS setup requested but setup file not found: $SETUP_FILE" >&2
    fi
fi
if [ "$ENABLE_ROS_BRIDGE" = "true" ]; then
    ROS_BRIDGE_ARGS+=(--ros-bridge)
    if [ "$ROS_PUBLISH_COMMANDS" != "true" ]; then
        ROS_BRIDGE_ARGS+=(--disable-command-publish)
    fi
fi
if [ "$ENABLE_CAMERA_BALL" = "true" ]; then
    CAMERA_ARGS+=(--camera-ball)
fi
if [ "$ENABLE_MOCAP_BALL" = "true" ]; then
    MOCAP_ARGS+=(--mocap-ball)
fi
if [ "$USE_HYBRID_DETECTOR" != "true" ]; then
    CAMERA_ARGS+=(--disable-hybrid-detector)
fi
if [ "$USE_DEPTH_SENSOR" != "true" ]; then
    CAMERA_ARGS+=(--disable-depth-sensor)
fi
if [ "$ENABLE_YOLO" = "true" ]; then
    CAMERA_ARGS+=(--enable-yolo)
fi
if [ "$ENABLE_HYBRID_CANDIDATE_FILTER" != "true" ]; then
    CAMERA_ARGS+=(--disable-hybrid-candidate-filter)
fi
if [ "$DETECTOR_HALF" = "true" ]; then
    CAMERA_ARGS+=(--detector-half)
fi
if [ "$USE_KALMAN" = "true" ]; then
    CAMERA_ARGS+=(--use-kalman)
fi
if [ "$USE_IR_DEPTH_KALMAN" = "true" ]; then
    CAMERA_ARGS+=(--use-ir-depth-kalman)
fi
if [ "$USE_SYNCHRONIZED_IR_DEPTH" = "true" ]; then
    CAMERA_ARGS+=(--use-synchronized-ir-depth)
fi
if [ "$USE_UNDISTORT" != "true" ]; then
    CAMERA_ARGS+=(--disable-undistort)
fi
if [ "$USE_ADAPTIVE_IR" != "true" ]; then
    CAMERA_ARGS+=(--disable-adaptive-ir)
fi
if [ "$ENABLE_ELLIPSE_FIT" != "true" ]; then
    CAMERA_ARGS+=(--disable-ellipse-fit)
fi
if [ "$ENABLE_TEMPORAL_TRACKER" = "true" ]; then
    CAMERA_ARGS+=(--enable-temporal-tracker)
fi
if [ "$ENABLE_CAMERA_DETECTION" != "true" ]; then
    CAMERA_ARGS+=(--disable-camera-detection)
fi
if [ "$ENABLE_CAMERA_VISUALIZATION" != "true" ]; then
    CAMERA_ARGS+=(--disable-camera-visualization)
fi
if [ "$ENABLE_CAMERA_COLOR_FILTER" != "true" ]; then
    CAMERA_ARGS+=(--disable-camera-color-filter)
fi

# The probe is a source-tree development tool.  Prefer the files being edited
# in this workspace over an older colcon install copy; otherwise safety-filter
# changes in rl_policy/compensation can silently be absent from a live run.
_PROBE_SOURCE_PYTHONPATH="$WORKSPACE_DIR/src/pingpong_moz1:$WORKSPACE_DIR/src/pingpong_replay"
export PYTHONPATH="$_PROBE_SOURCE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"

ANALYSIS_ARGS=()
select_analysis_recording_dir() {
    local mode="${ANALYSIS:-false}"
    local mode_lc="${mode,,}"
    local selected="${ANALYSIS_RUN_DIR:-}"
    local dirs=()
    local item
    local answer
    local idx
    if [ -z "$selected" ]; then
        case "$mode_lc" in
            ""|0|false|no|off|none)
                return 0
                ;;
            latest|true|yes|1|prompt|choose|select)
                if [ -d "$PROBE_RECORDING_OUTPUT_ROOT" ]; then
                    while IFS= read -r item; do
                        [ -n "$item" ] && dirs+=("$item")
                    done < <(
                        find "$PROBE_RECORDING_OUTPUT_ROOT" \
                            -mindepth 1 -maxdepth 1 -type d \
                            -name 'probe_recording_*' \
                            -printf '%T@ %p\n' 2>/dev/null \
                            | sort -nr | sed 's/^[^ ]* //'
                    )
                fi
                if [ "${#dirs[@]}" -eq 0 ]; then
                    echo "WARNING: ANALYSIS requested but no probe_recording_* dirs found under $PROBE_RECORDING_OUTPUT_ROOT" >&2
                    return 0
                fi
                if [ "$mode_lc" = "latest" ] || [ ! -t 0 ]; then
                    selected="${dirs[0]}"
                else
                    echo "[ProbeConfig] available analysis recordings:"
                    for idx in "${!dirs[@]}"; do
                        [ "$idx" -lt 20 ] || break
                        printf '  %2d) %s\n' "$((idx + 1))" "${dirs[$idx]}"
                    done
                    read -r -p "Select analysis recording [1] (or paste path): " answer
                    answer="${answer:-1}"
                    if [[ "$answer" =~ ^[0-9]+$ ]]; then
                        idx=$((answer - 1))
                        if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#dirs[@]}" ]; then
                            echo "ERROR: analysis selection out of range: $answer" >&2
                            exit 2
                        fi
                        selected="${dirs[$idx]}"
                    else
                        selected="$answer"
                    fi
                fi
                ;;
            *)
                selected="$mode"
                ;;
        esac
    fi
    if [ -n "$selected" ]; then
        if [ ! -d "$selected" ]; then
            echo "ERROR: analysis recording directory not found: $selected" >&2
            exit 2
        fi
        ANALYSIS_ARGS+=(--analysis-run-dir "$selected")
        echo "[ProbeConfig] analysis run=$selected"
    fi
}
select_analysis_recording_dir

exec python3 "$PROBE_SCRIPT" \
    --host "$HOST" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --policy-profile "$POLICY_PROFILE" \
    --simcode2-model-path "$SIMCODE2_MODEL_PATH" \
    --simcode2-policy-xml-path "$SIMCODE2_POLICY_XML_PATH" \
    --simcode3-model-path "$SIMCODE3_MODEL_PATH" \
    --simcode3-policy-xml-path "$SIMCODE3_POLICY_XML_PATH" \
    --simcode4-model-path "$SIMCODE4_MODEL_PATH" \
    --simcode4-policy-xml-path "$SIMCODE4_POLICY_XML_PATH" \
    --simcode5-model-path "$SIMCODE5_MODEL_PATH" \
    --simcode5-policy-xml-path "$SIMCODE5_POLICY_XML_PATH" \
    --robot-xml-path "$ROBOT_XML" \
    --policy-obs-dim "$POLICY_OBS_DIM" \
    --reset-initial-pose "$RESET_INITIAL_POSE" \
    --ball-lost-failsafe-timeout-s "$BALL_LOST_FAILSAFE_TIMEOUT_S" \
    --ball-release-min-downward-velocity-m-s "$BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S" \
    --ball-release-min-z-drop-m "$BALL_RELEASE_MIN_Z_DROP_M" \
    --ball-release-window-s "$BALL_RELEASE_WINDOW_S" \
    --ball-release-required-consecutive-ticks "$BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS" \
    --ball-release-max-obs-age-s "$BALL_RELEASE_MAX_OBS_AGE_S" \
    --done-z-min-m "$DONE_Z_MIN_M" \
    --done-z-max-m "$DONE_Z_MAX_M" \
    --done-x-range-m "$DONE_X_RANGE_M" \
    --done-y-range-m "$DONE_Y_RANGE_M" \
    --done-ball-required-consecutive-ticks "$DONE_BALL_REQUIRED_CONSECUTIVE_TICKS" \
    --done-racket-anchor-distance-m "$DONE_RACKET_ANCHOR_DISTANCE_M" \
    --done-racket-z-high-offset-m "$DONE_RACKET_Z_HIGH_OFFSET_M" \
    --done-racket-z-low-offset-m "$DONE_RACKET_Z_LOW_OFFSET_M" \
    --ee-x-limits-m="$DONE_EE_X_LIMITS_M" \
    --ee-y-limits-m="$DONE_EE_Y_LIMITS_M" \
    --ee-z-limits-m="$DONE_EE_Z_LIMITS_M" \
    --ee-psi-limits-rad="$DONE_EE_PSI_LIMITS_RAD" \
    --ee-b-limits-rad="$DONE_EE_B_LIMITS_RAD" \
    --ee-b-min-rad="$DONE_EE_B_MIN_RAD" \
    --ee-b-max-rad="$DONE_EE_B_MAX_RAD" \
    --ee-c-limits-rad="$DONE_EE_C_LIMITS_RAD" \
    --ee-c-min-rad="$DONE_EE_C_MIN_RAD" \
    --ee-c-max-rad="$DONE_EE_C_MAX_RAD" \
    --ee-orientation-reference-xyzw="$DONE_EE_ORIENTATION_REFERENCE_XYZW" \
    --ee-max-orientation-error-deg "$DONE_EE_MAX_ORIENTATION_ERROR_DEG" \
    --done-prediction-horizon-s "$DONE_PREDICTION_HORIZON_S" \
    --done-prediction-dt-s "$DONE_PREDICTION_DT_S" \
    --done-prediction-margin-m "$DONE_PREDICTION_MARGIN_M" \
    --done-prediction-orientation-margin-rad "$DONE_PREDICTION_ORIENTATION_MARGIN_RAD" \
    --workspace-soft-margin-m "$WORKSPACE_SOFT_MARGIN_M" \
    --workspace-soft-pull-speed-m-s "$WORKSPACE_SOFT_PULL_SPEED_M_S" \
    --workspace-soft-damping "$WORKSPACE_SOFT_DAMPING" \
    --workspace-soft-max-correction-speed-m-s "$WORKSPACE_SOFT_MAX_CORRECTION_SPEED_M_S" \
    --workspace-soft-dls-damping "$WORKSPACE_SOFT_DLS_DAMPING" \
    --workspace-soft-nullspace-damping "$WORKSPACE_SOFT_NULLSPACE_DAMPING" \
    --workspace-soft-posture-gain-s "$WORKSPACE_SOFT_POSTURE_GAIN_S" \
    --workspace-soft-boundary-joint-speed-scale "$WORKSPACE_SOFT_BOUNDARY_JOINT_SPEED_SCALE" \
    --workspace-hold-trigger-margin-m "$WORKSPACE_HOLD_TRIGGER_MARGIN_M" \
    --ball-noise-std-m "$BALL_NOISE_STD_M" \
    --control-rate-hz "$CONTROL_RATE_HZ" \
    --default-control-mode "$DEFAULT_CONTROL_MODE" \
    --racket-tracking-height-offset-m "$RACKET_TRACKING_HEIGHT_OFFSET_M" \
    --racket-tracking-position-gain-s "$RACKET_TRACKING_POSITION_GAIN_S" \
    --racket-tracking-dls-damping "$RACKET_TRACKING_DLS_DAMPING" \
    --racket-tracking-max-cartesian-speed-m-s "$RACKET_TRACKING_MAX_CARTESIAN_SPEED_M_S" \
    --racket-tracking-max-cartesian-acceleration-m-s2 "$RACKET_TRACKING_MAX_CARTESIAN_ACCELERATION_M_S2" \
    --racket-tracking-joint-velocity-scale "$RACKET_TRACKING_JOINT_VELOCITY_SCALE" \
    --racket-tracking-joint-acceleration-scale "$RACKET_TRACKING_JOINT_ACCELERATION_SCALE" \
    --racket-tracking-posture-gain-s "$RACKET_TRACKING_POSTURE_GAIN_S" \
    --racket-tracking-mpc-horizon-steps "$RACKET_TRACKING_MPC_HORIZON_STEPS" \
    --racket-tracking-mpc-position-weight "$RACKET_TRACKING_MPC_POSITION_WEIGHT" \
    --racket-tracking-mpc-target-velocity-weight "$RACKET_TRACKING_MPC_TARGET_VELOCITY_WEIGHT" \
    --racket-tracking-mpc-velocity-weight "$RACKET_TRACKING_MPC_VELOCITY_WEIGHT" \
    --racket-tracking-mpc-smooth-weight "$RACKET_TRACKING_MPC_SMOOTH_WEIGHT" \
    --racket-tracking-mpc-posture-weight "$RACKET_TRACKING_MPC_POSTURE_WEIGHT" \
    --racket-tracking-target-boundary-margin-m "$RACKET_TRACKING_TARGET_BOUNDARY_MARGIN_M" \
    --racket-tracking-max-ball-age-s "$RACKET_TRACKING_MAX_BALL_AGE_S" \
    --juggle-countdown-s "$JUGGLE_COUNTDOWN_S" \
    --juggle-target-apex-height-m "$JUGGLE_TARGET_APEX_HEIGHT_M" \
    --juggle-seed-apex-height-m "$JUGGLE_SEED_APEX_HEIGHT_M" \
    --juggle-apex-ramp-per-bounce-m "$JUGGLE_APEX_RAMP_PER_BOUNCE_M" \
    --juggle-launch-peak-acceleration-m-s2 "$JUGGLE_LAUNCH_PEAK_ACCELERATION_M_S2" \
    --juggle-restitution "$JUGGLE_RESTITUTION" \
    --juggle-tangential-transfer "$JUGGLE_TANGENTIAL_TRANSFER" \
    --juggle-drag-coefficient-per-m "$JUGGLE_DRAG_COEFFICIENT_PER_M" \
    --juggle-ball-radius-m "$JUGGLE_BALL_RADIUS_M" \
    --juggle-ready-xy-tolerance-m "$JUGGLE_READY_XY_TOLERANCE_M" \
    --juggle-ready-height-tolerance-m "$JUGGLE_READY_HEIGHT_TOLERANCE_M" \
    --juggle-ready-max-ball-speed-m-s "$JUGGLE_READY_MAX_BALL_SPEED_M_S" \
    --juggle-ready-settle-s "$JUGGLE_READY_SETTLE_S" \
    --juggle-max-ball-age-s "$JUGGLE_MAX_BALL_AGE_S" \
    --juggle-contact-grace-s "$JUGGLE_CONTACT_GRACE_S" \
    --safety-parking-velocity-scale "$SAFETY_PARKING_VELOCITY_SCALE" \
    --safety-parking-acceleration-scale "$SAFETY_PARKING_ACCELERATION_SCALE" \
    --compensation-mode "$COMPENSATION_MODE" \
    --comp-delay-ms "$COMP_DELAY_MS" \
    --real-comp-delay-ms "$REAL_COMP_DELAY_MS" \
    --comp-actuator-tau-s "$COMP_ACTUATOR_TAU_S" \
    --real-comp-actuator-tau-s "$REAL_COMP_ACTUATOR_TAU_S" \
    --comp-actuator-gain "$COMP_ACTUATOR_GAIN" \
    --real-comp-actuator-gain "$REAL_COMP_ACTUATOR_GAIN" \
    --comp-buffer-extra-steps "$COMP_BUFFER_EXTRA_STEPS" \
    --comp-mpc-beta "$COMP_MPC_BETA" \
    --real-comp-mpc-beta "$REAL_COMP_MPC_BETA" \
    --comp-mpc-delay-scale "$COMP_MPC_DELAY_SCALE" \
    --real-comp-mpc-delay-scale "$REAL_COMP_MPC_DELAY_SCALE" \
    --comp-mpc-tau-scale "$COMP_MPC_TAU_SCALE" \
    --real-comp-mpc-tau-scale "$REAL_COMP_MPC_TAU_SCALE" \
    --comp-mpc-horizon-steps "$COMP_MPC_HORIZON_STEPS" \
    --real-comp-mpc-horizon-steps "$REAL_COMP_MPC_HORIZON_STEPS" \
    --comp-mpc-tracking-weight "$COMP_MPC_TRACKING_WEIGHT" \
    --real-comp-mpc-tracking-weight "$REAL_COMP_MPC_TRACKING_WEIGHT" \
    --comp-mpc-nominal-weight "$COMP_MPC_NOMINAL_WEIGHT" \
    --real-comp-mpc-nominal-weight "$REAL_COMP_MPC_NOMINAL_WEIGHT" \
    --comp-mpc-delta-weight "$COMP_MPC_DELTA_WEIGHT" \
    --real-comp-mpc-delta-weight "$REAL_COMP_MPC_DELTA_WEIGHT" \
    --comp-mpc-max-delta-rad "$COMP_MPC_MAX_DELTA_RAD" \
    --real-comp-mpc-max-delta-rad "$REAL_COMP_MPC_MAX_DELTA_RAD" \
    --ball-observation-rate-hz "$BALL_OBSERVATION_RATE_HZ" \
    --backend-snapshot-rate-hz "$PROBE_BACKEND_SNAPSHOT_RATE_HZ" \
    --browser-step-rate-hz "$PROBE_STEP_RATE_HZ" \
    --display-rate-hz "$PROBE_DISPLAY_RATE_HZ" \
    --recording-output-root "$PROBE_RECORDING_OUTPUT_ROOT" \
    --recording-control-rate-hz "$PROBE_RECORDING_CONTROL_RATE_HZ" \
    --recording-image-rate-hz "$PROBE_RECORDING_IMAGE_RATE_HZ" \
    --recording-analysis-video-fps "$PROBE_RECORDING_ANALYSIS_VIDEO_FPS" \
    --recording-queue-size "$PROBE_RECORDING_QUEUE_SIZE" \
    "${ANALYSIS_ARGS[@]}" \
    --sim-backend "$SIM_BACKEND" \
    --sim-xml-path "$SIM_XML_PATH" \
    --simcode2-sim-xml-path "$SIMCODE2_SIM_XML_PATH" \
    --simcode3-sim-xml-path "$SIMCODE3_SIM_XML_PATH" \
    --simcode4-sim-xml-path "$SIMCODE4_SIM_XML_PATH" \
    --simcode5-sim-xml-path "$SIMCODE5_SIM_XML_PATH" \
    --sim-frame-rate-hz "$SIM_FRAME_RATE_HZ" \
    --sim-render-width "$SIM_RENDER_WIDTH" \
    --sim-render-height "$SIM_RENDER_HEIGHT" \
    --hybrid-ball-tracker-config-path "$HYBRID_BALL_TRACKER_CONFIG_PATH" \
    --ball-estimator-max-extrapolation-s "$BALL_ESTIMATOR_MAX_EXTRAPOLATION_S" \
    --ball-estimator-control-max-extrapolation-s "$BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S" \
    --ball-estimator-control-output-mode "$BALL_ESTIMATOR_CONTROL_OUTPUT_MODE" \
    --ball-estimator-frame-hold-output-rate-hz "$BALL_ESTIMATOR_FRAME_HOLD_OUTPUT_RATE_HZ" \
    --ball-estimator-trust-loss-s "$BALL_ESTIMATOR_TRUST_LOSS_S" \
    --ball-estimator-degrade-loss-s "$BALL_ESTIMATOR_DEGRADE_LOSS_S" \
    --ball-estimator-invalid-after-s "$BALL_ESTIMATOR_INVALID_AFTER_S" \
    --ball-estimator-prediction-horizon-s "$BALL_ESTIMATOR_PREDICTION_HORIZON_S" \
    --ball-estimator-q-pos-mm2-s "$BALL_ESTIMATOR_Q_POS_MM2_S" \
    --ball-estimator-q-vel-mult "$BALL_ESTIMATOR_Q_VEL_MULT" \
    --ball-estimator-q-depth-pos-mult "$BALL_ESTIMATOR_Q_DEPTH_POS_MULT" \
    --ball-estimator-q-depth-vel-mult "$BALL_ESTIMATOR_Q_DEPTH_VEL_MULT" \
    --ball-estimator-r-lateral-mm "$BALL_ESTIMATOR_R_LATERAL_MM" \
    --ball-estimator-r-depth-mm "$BALL_ESTIMATOR_R_DEPTH_MM" \
    --command-topic "$COMMAND_TOPIC" \
    --joint-state-topic "$JOINT_STATE_TOPIC" \
    --cart-state-topic "$CART_STATE_TOPIC" \
    --right-arm-mu-idx "$RIGHT_ARM_MU_IDX" \
    --camera-source-mode "$CAMERA_SOURCE_MODE" \
    --image-width "$IMAGE_WIDTH" \
    --image-height "$IMAGE_HEIGHT" \
    --realsense-fps "$REALSENSE_FPS" \
    --camera-image-topic "$CAMERA_IMAGE_TOPIC" \
    --camera-raw-image-topic "$CAMERA_RAW_IMAGE_TOPIC" \
    --camera-image-transport "$CAMERA_IMAGE_TRANSPORT" \
    --camera-info-topic "$CAMERA_INFO_TOPIC" \
    --calibration-json-path "$CALIBRATION_JSON_PATH" \
    --yolo-model-path "$YOLO_MODEL_PATH" \
    --target-class "$TARGET_CLASS" \
    --conf-threshold "$CONF_THRESHOLD" \
    --ir-params-json "$IR_PARAMS_JSON" \
    --yolo-interval "$YOLO_INTERVAL" \
    --candidate-min-brightness "$CANDIDATE_MIN_BRIGHTNESS" \
    --candidate-max-depth-disagreement-mm "$CANDIDATE_MAX_DEPTH_DISAGREEMENT_MM" \
    --candidate-motion-scale-px "$CANDIDATE_MOTION_SCALE_PX" \
    --ir-threshold "$IR_THRESHOLD" \
    --ir-exposure "$IR_EXPOSURE" \
    --ir-gain "$IR_GAIN" \
    --ir-laser-power "$IR_LASER_POWER" \
    --ir-min-radius "$IR_MIN_RADIUS" \
    --ir-max-radius "$IR_MAX_RADIUS" \
    --ir-circularity-min "$IR_CIRCULARITY_MIN" \
    --ir-brightness-percentile "$IR_BRIGHTNESS_PERCENTILE" \
    --hybrid-roi-margin "$HYBRID_ROI_MARGIN" \
    --depth-correction-model-path "$DEPTH_CORRECTION_MODEL_PATH" \
    --depth-neighborhood-px "$DEPTH_NEIGHBORHOOD_PX" \
    --depth-temporal-window-frames "$DEPTH_TEMPORAL_WINDOW_FRAMES" \
    --infer-imgsz "$INFER_IMGSZ" \
    --detector-device "$DETECTOR_DEVICE" \
    --camera-tilt-deg "$CAMERA_TILT_DEG" \
    --camera-visualization-rate-hz "$CAMERA_VISUALIZATION_RATE_HZ" \
    --camera-web-max-width "$CAMERA_WEB_MAX_WIDTH" \
    --camera-web-jpeg-quality "$CAMERA_WEB_JPEG_QUALITY" \
    --tracker-max-miss-bridge "$TRACKER_MAX_MISS_BRIDGE" \
    --tracker-gate-px "$TRACKER_GATE_PX" \
    --hsv-lower "$HSV_LOWER" \
    --hsv-upper "$HSV_UPPER" \
    --color-match-threshold "$COLOR_MATCH_THRESHOLD" \
    --color-circularity-min "$COLOR_CIRCULARITY_MIN" \
    --sim-cartesian-x-offset-m "$SIM_CARTESIAN_X_OFFSET_M" \
    --sim-cartesian-y-offset-m "$SIM_CARTESIAN_Y_OFFSET_M" \
    --sim-base-height-m "$SIM_BASE_HEIGHT_M" \
    "${FLOW_ARGS[@]}" \
    "${ROS_BRIDGE_ARGS[@]}" \
    "${CAMERA_ARGS[@]}" \
    "${MOCAP_ARGS[@]}" \
    --mocap-udp-host "$MOCAP_UDP_HOST" \
    --mocap-udp-port "$MOCAP_UDP_PORT" \
    --mocap-calibration-path "$MOCAP_CALIBRATION_PATH" \
    --mocap-marker-id "$MOCAP_MARKER_ID" \
    --mocap-rigid-body-slot "$MOCAP_RIGID_BODY_SLOT" \
    --mocap-rigid-body-id "$MOCAP_RIGID_BODY_ID" \
    --mocap-rigid-body-name "$MOCAP_RIGID_BODY_NAME" \
    --mocap-position-unit "$MOCAP_POSITION_UNIT" \
    --mocap-timestamp-unit "$MOCAP_TIMESTAMP_UNIT" \
    --mocap-timeout-s "$MOCAP_TIMEOUT_S" \
    --mocap-velocity-alpha "$MOCAP_VELOCITY_ALPHA" \
    --mocap-max-velocity-m-s "$MOCAP_MAX_VELOCITY_M_S" \
    "$@"
