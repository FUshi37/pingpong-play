#!/usr/bin/env bash
# Run one real-hardware pingpong policy experiment and generate diagnostics.

set -uo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/spirit-ai/pingpong_moz1}"
SETUP_FILE="${SETUP_FILE:-$WORKSPACE_DIR/install/setup.bash}"

if [ ! -f "$SETUP_FILE" ]; then
    echo "ERROR: ROS workspace setup file not found: $SETUP_FILE"
    echo "Build/source first, e.g.: colcon build --packages-select pingpong_controller"
    exit 1
fi

set +u
source "$SETUP_FILE"
set -u

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<EOF
Usage:
  ./RUN_REAL_POLICY_EXPERIMENT.sh [-- ROS_EXTRA_ARGS...]

Environment overrides:
  OUTPUT_ROOT=/path/to/run_dir
  RUN_DURATION_S=20.0
  INIT_POSE_DURATION_S=8.0
  ENABLE_VISUALIZATION=true
  DONE_Z_MIN_M=0.6
  MODEL_PATH=/path/to/policy.pkl-or-mjx-log-dir
  POLICY_OBS_DIM=67      # 67=inverse-MPC delay-conditioned model
  ENABLE_COMPENSATION=true
  ENABLE_BALL_RELEASE_GATE=true
  BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S=0.15
  BALL_RELEASE_MIN_Z_DROP_M=0.015

Extra ROS args after -- are appended last, so they can override defaults:
  ./RUN_REAL_POLICY_EXPERIMENT.sh -- -p run_duration_s:=30.0
EOF
    exit 0
fi

if [ "${1:-}" = "--" ]; then
    shift
fi
ROS_EXTRA_ARGS=("$@")

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_BASE:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/real_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE/real_run_$TIMESTAMP}"

PARAMS_FILE="${PARAMS_FILE:-$WORKSPACE_DIR/config_done_monitoring.yaml}"
IR_PARAMS_JSON="${IR_PARAMS_JSON:-$WORKSPACE_DIR/ir_params.json}"
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/models/best.pt}"
CALIBRATION_JSON_PATH="${CALIBRATION_JSON_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/vision_calib/eye_hand_result_848_undistorted.json}"
DEPTH_CORRECTION_MODEL_PATH="${DEPTH_CORRECTION_MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/depth_correction_model.pkl}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/inverse_compensation/18_stage4g_strong_contact_dr_inverse_mpc.pkl}"
ROBOT_XML="${ROBOT_XML:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml}"

COMMAND_TOPIC="${COMMAND_TOPIC:-/mx_mix_command}"
JOINT_STATE_TOPIC="${JOINT_STATE_TOPIC:-/joint_states}"
RIGHT_ARM_MU_IDX="${RIGHT_ARM_MU_IDX:-3}"
IMAGE_WIDTH="${IMAGE_WIDTH:-848}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
REALSENSE_FPS="${REALSENSE_FPS:-60}"
TARGET_CLASS="${TARGET_CLASS:-Class 0}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"

CONTROL_RATE_HZ="${CONTROL_RATE_HZ:-200.0}"
CONTROL_DT="$(python3 -c "print(1.0 / float('$CONTROL_RATE_HZ'))")"
RUN_DURATION_S="${RUN_DURATION_S:-20.0}"
INIT_POSE_DURATION_S="${INIT_POSE_DURATION_S:-8.0}"
POST_ENTER_START_DELAY_S="${POST_ENTER_START_DELAY_S:-3.0}"
REQUIRE_ENTER_CONFIRM="${REQUIRE_ENTER_CONFIRM:-true}"
ENABLE_VISUALIZATION="${ENABLE_VISUALIZATION:-true}"
ENABLE_SAFETY_LIMITER="${ENABLE_SAFETY_LIMITER:-false}"
ENABLE_COMPENSATION="${ENABLE_COMPENSATION:-true}"
POLICY_OBS_DIM="${POLICY_OBS_DIM:-67}"
ENABLE_BALL_RELEASE_GATE="${ENABLE_BALL_RELEASE_GATE:-true}"
BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S="${BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S:-0.15}"
BALL_RELEASE_MIN_Z_DROP_M="${BALL_RELEASE_MIN_Z_DROP_M:-0.015}"
BALL_RELEASE_WINDOW_S="${BALL_RELEASE_WINDOW_S:-0.12}"
BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS="${BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS:-3}"
BALL_RELEASE_MAX_OBS_AGE_S="${BALL_RELEASE_MAX_OBS_AGE_S:-0.08}"

DONE_Z_MIN_M="${DONE_Z_MIN_M:-0.6}"
DONE_Z_MAX_M="${DONE_Z_MAX_M:-1.9}"
DONE_X_RANGE_M="${DONE_X_RANGE_M:-0.5}"
DONE_Y_RANGE_M="${DONE_Y_RANGE_M:-0.5}"

SIM_CARTESIAN_X_OFFSET_M="${SIM_CARTESIAN_X_OFFSET_M:-0.0}"
SIM_CARTESIAN_Y_OFFSET_M="${SIM_CARTESIAN_Y_OFFSET_M:-0.023583}"
SIM_BASE_HEIGHT_M="${SIM_BASE_HEIGHT_M:-0.947396}"

TRAJECTORY_CSV="$OUTPUT_ROOT/ball_trajectory_base_link.csv"
TRAJECTORY_JSON="$OUTPUT_ROOT/ball_trajectory_base_link.json"
TRAJECTORY_NODE_PNG="$OUTPUT_ROOT/ball_trajectory_base_link_node.png"
POLICY_TRACE="$OUTPUT_ROOT/policy_obs_trace.csv"
ESTIMATOR_TRACE="$OUTPUT_ROOT/estimator_trace.csv"
RUN_LOG="$OUTPUT_ROOT/ros2_run.log"
TRAJECTORY_3D_PNG="$OUTPUT_ROOT/trajectory_3d.png"
TRAJECTORY_3D_HTML="$OUTPUT_ROOT/trajectory_3d.html"
REAL_RUN_METRICS_JSON="$OUTPUT_ROOT/real_run_metrics.json"
REAL_RUN_METRICS_REPORT="$OUTPUT_ROOT/real_run_metrics.txt"
POLICY_SAFETY_JSON="$OUTPUT_ROOT/policy_safety_summary.json"

REAL_2D_PLOTTER="$WORKSPACE_DIR/plot_real_run_2d.py"
TRAJECTORY_3D_PLOTTER="$WORKSPACE_DIR/plot_safety_replay_3d.py"
REAL_RUN_METRICS_ANALYZER="$WORKSPACE_DIR/analyze_real_run_metrics.py"
POLICY_SAFETY_ANALYZER="$WORKSPACE_DIR/analyze_policy_safety_trace.py"

mkdir -p "$OUTPUT_ROOT"

ROS_CONFIG_ARGS=()
if [ -f "$PARAMS_FILE" ]; then
    ROS_CONFIG_ARGS+=(--params-file "$PARAMS_FILE")
fi

cat > "$OUTPUT_ROOT/run_metadata.txt" <<EOF
timestamp=$TIMESTAMP
workspace=$WORKSPACE_DIR
output_root=$OUTPUT_ROOT
model_path=$MODEL_PATH
robot_xml=$ROBOT_XML
calibration_json_path=$CALIBRATION_JSON_PATH
ir_params_json=$IR_PARAMS_JSON
depth_correction_model_path=$DEPTH_CORRECTION_MODEL_PATH
run_duration_s=$RUN_DURATION_S
init_pose_duration_s=$INIT_POSE_DURATION_S
policy_obs_dim=$POLICY_OBS_DIM
control_rate_hz=$CONTROL_RATE_HZ
control_dt=$CONTROL_DT
enable_compensation=$ENABLE_COMPENSATION
enable_ball_release_gate=$ENABLE_BALL_RELEASE_GATE
ball_release_min_downward_velocity_m_s=$BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S
ball_release_min_z_drop_m=$BALL_RELEASE_MIN_Z_DROP_M
ball_release_window_s=$BALL_RELEASE_WINDOW_S
ball_release_required_consecutive_ticks=$BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS
ball_release_max_obs_age_s=$BALL_RELEASE_MAX_OBS_AGE_S
done_z_min_m=$DONE_Z_MIN_M
sim_offsets=[$SIM_CARTESIAN_X_OFFSET_M, $SIM_CARTESIAN_Y_OFFSET_M, $SIM_BASE_HEIGHT_M]
EOF

echo "========================================================================"
echo "Real pingpong policy experiment"
echo "========================================================================"
echo "Output root: $OUTPUT_ROOT"
echo "Policy trace: $POLICY_TRACE"
echo "Estimator trace: $ESTIMATOR_TRACE"
echo "Trajectory CSV: $TRAJECTORY_CSV"
echo "3D HTML: $TRAJECTORY_3D_HTML"
echo ""

set +e
ros2 run pingpong_controller pingpong_node --ros-args \
    "${ROS_CONFIG_ARGS[@]}" \
    -p command_topic:="$COMMAND_TOPIC" \
    -p joint_state_topic:="$JOINT_STATE_TOPIC" \
    -p right_arm_mu_idx:="$RIGHT_ARM_MU_IDX" \
    -p use_direct_realsense:=true \
    -p use_hybrid_detector:=true \
    -p ir_params_json:="$IR_PARAMS_JSON" \
    -p yolo_model_path:="$YOLO_MODEL_PATH" \
    -p calibration_json_path:="$CALIBRATION_JSON_PATH" \
    -p image_width:="$IMAGE_WIDTH" \
    -p image_height:="$IMAGE_HEIGHT" \
    -p realsense_fps:="$REALSENSE_FPS" \
    -p target_class:="$TARGET_CLASS" \
    -p conf_threshold:="$CONF_THRESHOLD" \
    -p color_match_threshold:=0.0 \
    -p enable_rl_policy:=true \
    -p policy_backend:=mjx \
    -p rl_model_path:="$MODEL_PATH" \
    -p policy_obs_dim:="$POLICY_OBS_DIM" \
    -p robot_xml_path:="$ROBOT_XML" \
    -p control_rate_hz:="$CONTROL_RATE_HZ" \
    -p enable_init_pose:=true \
    -p init_pose_duration_s:="$INIT_POSE_DURATION_S" \
    -p require_enter_confirm:="$REQUIRE_ENTER_CONFIRM" \
    -p post_enter_start_delay_s:="$POST_ENTER_START_DELAY_S" \
    -p run_duration_s:="$RUN_DURATION_S" \
    -p shutdown_after_run_duration:=true \
    -p shutdown_after_run_duration_delay_s:=1.0 \
    -p enable_safety_limiter:="$ENABLE_SAFETY_LIMITER" \
    -p enable_compensation:="$ENABLE_COMPENSATION" \
    -p enable_ball_lost_failsafe:=true \
    -p enable_ball_capture_gate:=true \
    -p enable_ball_release_gate:="$ENABLE_BALL_RELEASE_GATE" \
    -p ball_release_min_downward_velocity_m_s:="$BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S" \
    -p ball_release_min_z_drop_m:="$BALL_RELEASE_MIN_Z_DROP_M" \
    -p ball_release_window_s:="$BALL_RELEASE_WINDOW_S" \
    -p ball_release_required_consecutive_ticks:="$BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS" \
    -p ball_release_max_obs_age_s:="$BALL_RELEASE_MAX_OBS_AGE_S" \
    -p enable_ball_state_estimator:=true \
    -p use_depth_sensor:=true \
    -p enable_visualization:="$ENABLE_VISUALIZATION" \
    -p start_confirm_topic:=/pingpong/start_control \
    -p auto_continue_on_stdin_eof:=false \
    -p enter_prompt_log_period_s:=5.0 \
    -p quiet_vision_logs:=true \
    -p print_every:=0 \
    -p depth_correction_model_path:="$DEPTH_CORRECTION_MODEL_PATH" \
    -p sim_cartesian_x_offset_m:="$SIM_CARTESIAN_X_OFFSET_M" \
    -p sim_cartesian_y_offset_m:="$SIM_CARTESIAN_Y_OFFSET_M" \
    -p sim_base_height_m:="$SIM_BASE_HEIGHT_M" \
    -p done_ball_z_floor_m:="$DONE_Z_MIN_M" \
    -p done_ball_z_ceiling_m:="$DONE_Z_MAX_M" \
    -p done_ball_x_range_m:="$DONE_X_RANGE_M" \
    -p done_ball_y_range_m:="$DONE_Y_RANGE_M" \
    -p save_trajectory_csv:=true \
    -p trajectory_csv_path:="$TRAJECTORY_CSV" \
    -p save_trajectory_json:=true \
    -p trajectory_json_path:="$TRAJECTORY_JSON" \
    -p save_trajectory_plot:=true \
    -p trajectory_plot_path:="$TRAJECTORY_NODE_PNG" \
    -p save_estimator_log:=true \
    -p estimator_log_path:="$ESTIMATOR_TRACE" \
    -p save_policy_log:=true \
    -p policy_log_path:="$POLICY_TRACE" \
    -p publish_rl_joint_cmd_debug:=true \
    "${ROS_EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
ROS_STATUS="${PIPESTATUS[0]}"
set -e

echo ""
echo "========================================================================"
echo "Post-processing real run"
echo "========================================================================"
echo "ros2 exit status: $ROS_STATUS"

if [ -f "$REAL_2D_PLOTTER" ]; then
    python3 "$REAL_2D_PLOTTER" \
        --trajectory-csv "$TRAJECTORY_CSV" \
        --policy-trace "$POLICY_TRACE" \
        --estimator-trace "$ESTIMATOR_TRACE" \
        --output-dir "$OUTPUT_ROOT" \
        --sim-base-height-m "$SIM_BASE_HEIGHT_M" \
        --sim-cartesian-x-offset-m "$SIM_CARTESIAN_X_OFFSET_M" \
        --sim-cartesian-y-offset-m "$SIM_CARTESIAN_Y_OFFSET_M" \
        --done-z-min-m "$DONE_Z_MIN_M" \
        --done-z-max-m "$DONE_Z_MAX_M"
else
    echo "WARNING: 2D plotter not found: $REAL_2D_PLOTTER"
fi

if [ -f "$TRAJECTORY_3D_PLOTTER" ] && [ -f "$TRAJECTORY_CSV" ] && [ -f "$POLICY_TRACE" ]; then
    python3 "$TRAJECTORY_3D_PLOTTER" \
        --replay-csv "$TRAJECTORY_CSV" \
        --policy-trace "$POLICY_TRACE" \
        --output "$TRAJECTORY_3D_PNG" \
        --html-output "$TRAJECTORY_3D_HTML" \
        --scenario "real_hardware_$TIMESTAMP" \
        --replay-frame base_link \
        --sim-base-height-m "$SIM_BASE_HEIGHT_M" \
        --sim-cartesian-x-offset-m "$SIM_CARTESIAN_X_OFFSET_M" \
        --sim-cartesian-y-offset-m "$SIM_CARTESIAN_Y_OFFSET_M" \
        --done-z-min-m "$DONE_Z_MIN_M" \
        --done-z-max-m "$DONE_Z_MAX_M" \
        --done-x-range-m "$DONE_X_RANGE_M" \
        --done-y-range-m "$DONE_Y_RANGE_M"
else
    echo "WARNING: skipped 3D plot; missing trajectory or policy trace"
fi

if [ -f "$REAL_RUN_METRICS_ANALYZER" ] && [ -f "$POLICY_TRACE" ]; then
    echo ""
    echo "Generating four-layer real-run diagnostics"
    python3 "$REAL_RUN_METRICS_ANALYZER" \
        --trajectory-csv "$TRAJECTORY_CSV" \
        --policy-trace "$POLICY_TRACE" \
        --estimator-trace "$ESTIMATOR_TRACE" \
        --run-log "$RUN_LOG" \
        --output-json "$REAL_RUN_METRICS_JSON" \
        --output-report "$REAL_RUN_METRICS_REPORT" \
        --dt "$CONTROL_DT" \
        || echo "WARNING: four-layer diagnostics failed"
else
    echo "WARNING: skipped four-layer diagnostics; missing analyzer or policy trace"
fi

if [ -f "$POLICY_SAFETY_ANALYZER" ] && [ -f "$POLICY_TRACE" ]; then
    echo ""
    echo "Generating policy safety trace summary"
    python3 "$POLICY_SAFETY_ANALYZER" "$POLICY_TRACE" \
        --output-json "$POLICY_SAFETY_JSON" \
        --dt "$CONTROL_DT" \
        --done-z-min-m "$DONE_Z_MIN_M" \
        --done-z-max-m "$DONE_Z_MAX_M" \
        --done-x-range-m "$DONE_X_RANGE_M" \
        --done-y-range-m "$DONE_Y_RANGE_M" \
        || echo "WARNING: policy safety trace summary failed"
else
    echo "WARNING: skipped policy safety trace summary; missing analyzer or policy trace"
fi

echo ""
echo "========================================================================"
echo "Real experiment complete"
echo "========================================================================"
echo "Output root: $OUTPUT_ROOT"
echo "Four-layer metrics: $REAL_RUN_METRICS_REPORT"
echo "Four-layer metrics JSON: $REAL_RUN_METRICS_JSON"
echo "Policy safety JSON: $POLICY_SAFETY_JSON"
echo "Ball 2D: $OUTPUT_ROOT/ball_state_2d.png"
echo "Policy I/O 2D: $OUTPUT_ROOT/policy_io_2d.png"
echo "3D HTML: $TRAJECTORY_3D_HTML"
echo "ROS log: $RUN_LOG"

exit "$ROS_STATUS"
