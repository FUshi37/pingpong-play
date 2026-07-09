#!/bin/bash
# Generate synthetic ball-state replay trajectories, run pingpong_node in replay
# mode, and write a per-scenario safety summary.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/spirit-ai/pingpong_moz1}"
SETUP_FILE="${SETUP_FILE:-$WORKSPACE_DIR/install/setup.bash}"

if [ ! -f "$SETUP_FILE" ]; then
    echo "ERROR: ROS workspace setup file not found: $SETUP_FILE"
    echo "Build/source the workspace first, e.g. colcon build --packages-select pingpong_controller"
    exit 1
fi

# ROS/colcon setup scripts may read optional environment variables such as
# COLCON_TRACE while they are unset. Temporarily relax nounset only while
# sourcing the generated environment.
set +u
source "$SETUP_FILE"
set -u

GENERATOR="$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/tools/generate_safety_replay_trajectories.py"
SAFETY_ANALYZER="$WORKSPACE_DIR/analyze_policy_safety_trace.py"
TRAJECTORY_PLOTTER="$WORKSPACE_DIR/plot_safety_replay_3d.py"
DASHBOARD_PLOTTER="$WORKSPACE_DIR/plot_safety_replay_dashboard.py"

SCENARIO="${SCENARIO:-all}"
if [ "$#" -gt 0 ]; then
    case "$1" in
        --)
            shift
            ;;
        --*)
            ;;
        *)
            SCENARIO="$1"
            shift
            ;;
    esac
fi
if [ "${1:-}" = "--" ]; then
    shift
fi
GENERATOR_EXTRA_ARGS=("$@")

if [ "$SCENARIO" = "all" ]; then
    SCENARIOS=(
        stationary
        jitter
        lost
        lost_after_seen
        lost_high_jitter_juggle
        exit_x_pos
        exit_x_neg
        exit_y_pos
        exit_y_neg
        exit_z_high
        exit_z_low
        jump
        up_occlusion
        normal_juggle
        wide_juggle
        near_floor_juggle
        near_ceiling_juggle
        edge_x_pos_hold
        edge_x_neg_hold
        edge_y_pos_hold
        edge_y_neg_hold
        edge_z_high_hold
        edge_z_low_hold
        fast_lateral_x
        fast_lateral_y
        side_arc
        spiral
        random_walk
        intermittent_occlusion
        burst_dropout
        flicker
        reacquire_shift
        false_positive_spike
        depth_spike
        velocity_spike
        drop_from_high
        low_fast_drop
        exit_xy_pos_pos
        exit_xy_pos_neg
        exit_xy_neg_pos
        exit_xy_neg_neg
    )
else
    IFS=',' read -r -a SCENARIOS <<< "$SCENARIO"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${OUTPUT_BASE:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/safety_replay}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE/safety_replay_$TIMESTAMP}"
DURATION_S="${DURATION_S:-6.0}"
LOST_HIGH_JITTER_JUGGLE_DURATION_S="${LOST_HIGH_JITTER_JUGGLE_DURATION_S:-12.0}"
RATE_HZ="${RATE_HZ:-60.0}"
FRAME="${FRAME:-sim}"

MODEL_PATH="${MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/inverse_compensation/18_stage4g_strong_contact_dr_inverse_mpc.pkl}"
POLICY_OBS_DIM="${POLICY_OBS_DIM:-67}"  # 67=inverse-MPC delay-conditioned model
ROBOT_XML="${ROBOT_XML:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml}"
PARAMS_FILE="${PARAMS_FILE:-$WORKSPACE_DIR/config_done_monitoring.yaml}"
REFERENCE_OBS="${REFERENCE_OBS:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/data/rl_sim/validate_obs_trace/validate_obs_trace.csv}"
REFERENCE_ACTION="${REFERENCE_ACTION:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/data/rl_sim/validate_obs_trace/validate_trace.csv}"

ENABLE_BALL_STATE_ESTIMATOR="${ENABLE_BALL_STATE_ESTIMATOR:-true}"
ENABLE_BALL_CAPTURE_GATE="${ENABLE_BALL_CAPTURE_GATE:-true}"
ENABLE_BALL_LOST_FAILSAFE="${ENABLE_BALL_LOST_FAILSAFE:-true}"
ENABLE_DONE_MONITORING="${ENABLE_DONE_MONITORING:-true}"
# Safety limiter is intentionally forced off for this replay workflow because
# the current limiter path is known to be buggy in the real-system simulator.
ENABLE_SAFETY_LIMITER="false"
# The default model is the 67D inverse-MPC deployment policy, so actuator
# compensation is part of the intended validation path.
ENABLE_COMPENSATION="${ENABLE_COMPENSATION:-true}"
REQUIRE_ENTER_CONFIRM="${REQUIRE_ENTER_CONFIRM:-true}"
PROMPT_BETWEEN_SCENARIOS="${PROMPT_BETWEEN_SCENARIOS:-true}"
RUN_REFERENCE_ANALYSIS="${RUN_REFERENCE_ANALYSIS:-false}"
PLOT_3D_TRAJECTORY="${PLOT_3D_TRAJECTORY:-true}"
PLOT_SCENARIO_PNG="${PLOT_SCENARIO_PNG:-false}"

DONE_Z_MIN_M="${DONE_Z_MIN_M:-0.8}"
DONE_Z_MAX_M="${DONE_Z_MAX_M:-1.9}"
DONE_X_RANGE_M="${DONE_X_RANGE_M:-0.5}"
DONE_Y_RANGE_M="${DONE_Y_RANGE_M:-0.5}"
DONE_RACKET_ANCHOR_DISTANCE_M="${DONE_RACKET_ANCHOR_DISTANCE_M:-1.1}"
DONE_RACKET_Z_HIGH_OFFSET_M="${DONE_RACKET_Z_HIGH_OFFSET_M:-0.24}"
DONE_RACKET_Z_LOW_OFFSET_M="${DONE_RACKET_Z_LOW_OFFSET_M:--0.12}"
DONE_EE_X_LIMITS_M="${DONE_EE_X_LIMITS_M:--0.429,-0.110}"
DONE_EE_Y_LIMITS_M="${DONE_EE_Y_LIMITS_M:--0.578,-0.293}"
DONE_EE_Z_LIMITS_M="${DONE_EE_Z_LIMITS_M:-1.050,1.430}"

if [ -n "$DONE_EE_X_LIMITS_M" ]; then
    ROS_DONE_EE_X_LIMITS_M="[$DONE_EE_X_LIMITS_M]"
else
    ROS_DONE_EE_X_LIMITS_M="[]"
fi
if [ -n "$DONE_EE_Y_LIMITS_M" ]; then
    ROS_DONE_EE_Y_LIMITS_M="[$DONE_EE_Y_LIMITS_M]"
else
    ROS_DONE_EE_Y_LIMITS_M="[]"
fi
if [ -n "$DONE_EE_Z_LIMITS_M" ]; then
    ROS_DONE_EE_Z_LIMITS_M="[$DONE_EE_Z_LIMITS_M]"
else
    ROS_DONE_EE_Z_LIMITS_M="[]"
fi

if [ "$FRAME" = "sim" ]; then
    REPLAY_IS_SIM_FRAME="true"
else
    REPLAY_IS_SIM_FRAME="false"
fi

ROS_CONFIG_ARGS=()
if [ -f "$PARAMS_FILE" ]; then
    ROS_CONFIG_ARGS+=(--params-file "$PARAMS_FILE")
fi

mkdir -p "$OUTPUT_ROOT"

echo "========================================================================"
echo "Policy safety scenario replay"
echo "========================================================================"
echo "Workspace: $WORKSPACE_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Scenarios: ${SCENARIOS[*]}"
echo "Frame: $FRAME (replay_data_is_sim_frame=$REPLAY_IS_SIM_FRAME)"
echo "Duration: ${DURATION_S}s"
echo "lost_high_jitter_juggle duration: ${LOST_HIGH_JITTER_JUGGLE_DURATION_S}s"
echo "Rate: ${RATE_HZ}Hz"
echo "Model path: $MODEL_PATH"
echo "Policy obs dim: $POLICY_OBS_DIM (0=auto from MJX checkpoint)"
echo "Estimator: $ENABLE_BALL_STATE_ESTIMATOR"
echo "Ball capture gate: $ENABLE_BALL_CAPTURE_GATE"
echo "Safety limiter: $ENABLE_SAFETY_LIMITER"
echo "Actuator compensation: $ENABLE_COMPENSATION"
echo "3D dashboard HTML: $PLOT_3D_TRAJECTORY"
echo "Per-scenario PNG: $PLOT_SCENARIO_PNG"
echo "EE workspace limits: x=$ROS_DONE_EE_X_LIMITS_M y=$ROS_DONE_EE_Y_LIMITS_M z=$ROS_DONE_EE_Z_LIMITS_M"
echo "Prompt between scenarios: $PROMPT_BETWEEN_SCENARIOS"
echo ""

SCENE_TOTAL="${#SCENARIOS[@]}"
SCENE_INDEX=0

for SC in "${SCENARIOS[@]}"; do
    SCENE_INDEX=$((SCENE_INDEX + 1))
    SC_OUTPUT_DIR="$OUTPUT_ROOT/$SC"
    REPLAY_CSV="$OUTPUT_ROOT/$SC.csv"
    POLICY_TRACE="$SC_OUTPUT_DIR/policy_obs_trace.csv"
    ESTIMATOR_TRACE="$SC_OUTPUT_DIR/estimator_trace.csv"
    TRAJECTORY_PLOT="$SC_OUTPUT_DIR/trajectory_3d.png"
    SC_DURATION_S="$DURATION_S"
    if [ "$SC" = "lost_high_jitter_juggle" ]; then
        SC_DURATION_S="$LOST_HIGH_JITTER_JUGGLE_DURATION_S"
    fi

    mkdir -p "$SC_OUTPUT_DIR"

    echo "========================================================================"
    echo "Generating scenario: $SC ($SCENE_INDEX/$SCENE_TOTAL)"
    echo "========================================================================"

    python3 "$GENERATOR" \
        --scenario "$SC" \
        --duration "$SC_DURATION_S" \
        --rate-hz "$RATE_HZ" \
        --frame "$FRAME" \
        --output "$REPLAY_CSV" \
        --done-z-min-m "$DONE_Z_MIN_M" \
        --done-z-max-m "$DONE_Z_MAX_M" \
        --done-x-range-m "$DONE_X_RANGE_M" \
        --done-y-range-m "$DONE_Y_RANGE_M" \
        "${GENERATOR_EXTRA_ARGS[@]}"

    echo ""
    echo "========================================================================"
    echo "Running scenario: $SC ($SCENE_INDEX/$SCENE_TOTAL)"
    echo "========================================================================"
    echo "Replay CSV: $REPLAY_CSV"
    echo "Policy trace: $POLICY_TRACE"
    echo ""

    ros2 run pingpong_controller pingpong_node --ros-args \
        "${ROS_CONFIG_ARGS[@]}" \
        -p enable_rl_policy:=true \
        -p policy_backend:=mjx \
        -p rl_model_path:="$MODEL_PATH" \
        -p policy_obs_dim:="$POLICY_OBS_DIM" \
        -p robot_xml_path:="$ROBOT_XML" \
        -p control_rate_hz:=200.0 \
        -p enable_replay_mode:=true \
        -p replay_csv_path:="$REPLAY_CSV" \
        -p replay_data_is_sim_frame:="$REPLAY_IS_SIM_FRAME" \
        -p replay_loop:=false \
        -p replay_auto_exit:=true \
        -p enable_init_pose:=true \
        -p init_pose_duration_s:=3.0 \
        -p require_enter_confirm:="$REQUIRE_ENTER_CONFIRM" \
        -p enable_ball_capture_gate:="$ENABLE_BALL_CAPTURE_GATE" \
        -p enable_ball_lost_failsafe:="$ENABLE_BALL_LOST_FAILSAFE" \
        -p enable_done_monitoring:="$ENABLE_DONE_MONITORING" \
        -p enable_ball_state_estimator:="$ENABLE_BALL_STATE_ESTIMATOR" \
        -p enable_safety_limiter:="$ENABLE_SAFETY_LIMITER" \
        -p enable_compensation:="$ENABLE_COMPENSATION" \
        -p compensation_mode:=inverse_mpc \
        -p comp_delay_ms:=72.0 \
        -p comp_actuator_tau_s:=0.074 \
        -p comp_mpc_beta:=1.2 \
        -p comp_mpc_delay_scale:=1.05 \
        -p comp_mpc_tau_scale:=0.75 \
        -p comp_mpc_horizon_steps:=6 \
        -p publish_rl_joint_cmd_debug:=true \
        -p right_arm_mu_idx:=3 \
        -p save_policy_log:=true \
        -p policy_log_path:="$POLICY_TRACE" \
        -p save_estimator_log:=true \
        -p estimator_log_path:="$ESTIMATOR_TRACE" \
        -p done_ball_z_floor_m:="$DONE_Z_MIN_M" \
        -p done_ball_z_ceiling_m:="$DONE_Z_MAX_M" \
        -p done_ball_x_range_m:="$DONE_X_RANGE_M" \
        -p done_ball_y_range_m:="$DONE_Y_RANGE_M" \
        -p done_racket_anchor_distance_m:="$DONE_RACKET_ANCHOR_DISTANCE_M" \
        -p done_racket_z_high_offset_m:="$DONE_RACKET_Z_HIGH_OFFSET_M" \
        -p done_racket_z_low_offset_m:="$DONE_RACKET_Z_LOW_OFFSET_M" \
        -p done_ee_x_limits_m:="$ROS_DONE_EE_X_LIMITS_M" \
        -p done_ee_y_limits_m:="$ROS_DONE_EE_Y_LIMITS_M" \
        -p done_ee_z_limits_m:="$ROS_DONE_EE_Z_LIMITS_M"

    echo ""
    echo "========================================================================"
    echo "Safety summary: $SC"
    echo "========================================================================"

    if [ -f "$POLICY_TRACE" ]; then
        python3 "$SAFETY_ANALYZER" \
            "$POLICY_TRACE" \
            --output-json "$SC_OUTPUT_DIR/safety_summary.json" \
            --done-z-min-m "$DONE_Z_MIN_M" \
            --done-z-max-m "$DONE_Z_MAX_M" \
            --done-x-range-m "$DONE_X_RANGE_M" \
            --done-y-range-m "$DONE_Y_RANGE_M" \
            --done-racket-anchor-distance-m "$DONE_RACKET_ANCHOR_DISTANCE_M" \
            --done-racket-z-high-offset-m "$DONE_RACKET_Z_HIGH_OFFSET_M" \
            --done-racket-z-low-offset-m "$DONE_RACKET_Z_LOW_OFFSET_M" \
            --ee-x-limits-m="$DONE_EE_X_LIMITS_M" \
            --ee-y-limits-m="$DONE_EE_Y_LIMITS_M" \
            --ee-z-limits-m="$DONE_EE_Z_LIMITS_M"
    else
        echo "WARNING: Policy trace not found: $POLICY_TRACE"
    fi

    if [ "$RUN_REFERENCE_ANALYSIS" = "true" ] && [ -f "$POLICY_TRACE" ]; then
        python3 "$WORKSPACE_DIR/analyze_policy_performance.py" \
            "$POLICY_TRACE" \
            --reference-obs "$REFERENCE_OBS" \
            --reference-action "$REFERENCE_ACTION" \
            --output-dir "$SC_OUTPUT_DIR/reference_comparison"
    fi

    if [ "$PLOT_SCENARIO_PNG" = "true" ] && [ -f "$POLICY_TRACE" ]; then
        echo ""
        echo "Generating per-scenario static 3D PNG..."
        python3 "$TRAJECTORY_PLOTTER" \
            --replay-csv "$REPLAY_CSV" \
            --policy-trace "$POLICY_TRACE" \
            --output "$TRAJECTORY_PLOT" \
            --scenario "$SC" \
            --replay-frame "$FRAME" \
            --done-z-min-m "$DONE_Z_MIN_M" \
            --done-z-max-m "$DONE_Z_MAX_M" \
            --done-x-range-m "$DONE_X_RANGE_M" \
            --done-y-range-m "$DONE_Y_RANGE_M" \
            --ee-x-limits-m="$DONE_EE_X_LIMITS_M" \
            --ee-y-limits-m="$DONE_EE_Y_LIMITS_M" \
            --ee-z-limits-m="$DONE_EE_Z_LIMITS_M"
    fi

    echo ""
    if [ "$PROMPT_BETWEEN_SCENARIOS" = "true" ] && [ "$SCENE_INDEX" -lt "$SCENE_TOTAL" ]; then
        NEXT_SC="${SCENARIOS[$SCENE_INDEX]}"
        echo "========================================================================"
        echo "Scenario '$SC' finished and node has stopped."
        echo "Next scenario: '$NEXT_SC'"
        echo "Press Enter once here to launch the next node; it will first restore init pose."
        echo "After init pose is reached, press Enter again in pingpong_node to start replay."
        echo "========================================================================"
        read -r -p "Press Enter to restore init pose for next scenario..." _
        echo ""
    fi
done

if [ "$PLOT_3D_TRAJECTORY" = "true" ]; then
    echo "========================================================================"
    echo "Generating single interactive 3D dashboard HTML"
    echo "========================================================================"
    python3 "$DASHBOARD_PLOTTER" \
        --root "$OUTPUT_ROOT" \
        --output "$OUTPUT_ROOT/safety_replay_dashboard.html" \
        --scenarios "$(IFS=','; echo "${SCENARIOS[*]}")" \
        --replay-frame "$FRAME" \
        --done-z-min-m "$DONE_Z_MIN_M" \
        --done-z-max-m "$DONE_Z_MAX_M" \
        --done-x-range-m "$DONE_X_RANGE_M" \
        --done-y-range-m "$DONE_Y_RANGE_M" \
        --ee-x-limits-m="$DONE_EE_X_LIMITS_M" \
        --ee-y-limits-m="$DONE_EE_Y_LIMITS_M" \
        --ee-z-limits-m="$DONE_EE_Z_LIMITS_M"
    echo ""
fi

echo "========================================================================"
echo "Safety scenario replay complete"
echo "========================================================================"
echo "Results saved to: $OUTPUT_ROOT"
if [ "$PLOT_3D_TRAJECTORY" = "true" ]; then
    echo "Interactive dashboard: $OUTPUT_ROOT/safety_replay_dashboard.html"
fi
