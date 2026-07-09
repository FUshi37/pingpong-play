#!/bin/bash
# Run pingpong_node in replay mode with policy logging enabled, then analyze results

set -e

# Source ROS2 workspace
source /home/spirit-ai/pingpong_moz1/install/setup.bash

# Configuration
# REPLAY_CSV: Path to ball trajectory CSV file
# REPLAY_IS_SIM_FRAME: Set to "true" if CSV is already in simulation coordinates (mjx_ball_replay.csv)
#                      Set to "false" if CSV is in real robot coordinates (needs transformation)

# Option 1: Pure simulation data (already in sim frame, NO transformation needed)
#REPLAY_CSV="/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/mjx_ball_replay.csv"
#REPLAY_IS_SIM_FRAME="true"

# Option 2: Real ball states (in base_link frame, transformation needed)
DETECTION_CSV="/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260706_205810_hybrid_annotatedv1_detections.csv"
REPLAY_DURATION_S=12.0
REPLAY_CSV="/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/simulation_replay/pingpong_raw_20260706_205810_replay_12s.csv"
REPLAY_IS_SIM_FRAME="false"


MODEL_PATH="/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/inverse_compensation/18_stage4g_strong_contact_dr_inverse_mpc.pkl"
ROBOT_XML="/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml"
OUTPUT_DIR="/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/simulation_replay"
REFERENCE_OBS="/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/data/rl_sim/validate_obs_trace/validate_obs_trace.csv"
REFERENCE_ACTION="/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/data/rl_sim/validate_obs_trace/validate_trace.csv"

if [ ! -f "$DETECTION_CSV" ]; then
    echo "ERROR: Detection CSV not found: $DETECTION_CSV"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Preparing replay CSV from offline hybrid detections..."
echo "========================================================================"
echo "Detection CSV: $DETECTION_CSV"
echo "Replay CSV: $REPLAY_CSV"
echo "Replay duration: ${REPLAY_DURATION_S}s"
echo ""

python3 /home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/tools/prepare_replay_data.py \
    "$DETECTION_CSV" \
    --duration "$REPLAY_DURATION_S" \
    --output "$REPLAY_CSV"

echo ""
echo "========================================================================"
echo "Running pingpong_node in replay mode with policy logging..."
echo "========================================================================"
echo "Replay CSV: $REPLAY_CSV"
echo "Replay is sim frame: $REPLAY_IS_SIM_FRAME"
echo "Model: $(basename $MODEL_PATH)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the node with policy logging enabled
ros2 run pingpong_controller pingpong_node --ros-args \
    -p enable_rl_policy:=true \
    -p policy_backend:=mjx \
    -p rl_model_path:=$MODEL_PATH \
    -p enable_delay_conditioning:=true \
    -p robot_xml_path:=$ROBOT_XML \
    -p control_rate_hz:=200.0 \
    -p enable_replay_mode:=true \
    -p replay_csv_path:=$REPLAY_CSV \
    -p replay_data_is_sim_frame:=$REPLAY_IS_SIM_FRAME \
    -p replay_loop:=false \
    -p enable_init_pose:=true \
    -p init_pose_duration_s:=3.0 \
    -p require_enter_confirm:=true \
    -p enable_ball_lost_failsafe:=false \
    -p enable_ball_state_estimator:=false \
    -p enable_safety_limiter:=false \
    -p enable_compensation:=true \
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
    -p policy_log_path:=$OUTPUT_DIR/policy_obs_trace.csv \
    -p params_file:=config_done_monitoring.yaml

echo ""
echo "========================================================================"
echo "Node execution completed. Analyzing results..."
echo "========================================================================"
echo ""

# Policy trace file (fixed name)
POLICY_TRACE="$OUTPUT_DIR/policy_obs_trace.csv"

if [ ! -f "$POLICY_TRACE" ]; then
    echo "ERROR: Policy trace not found: $POLICY_TRACE"
    exit 1
fi

echo "Found policy trace: $POLICY_TRACE"
echo "Reference obs trace: $REFERENCE_OBS"
echo "Reference action trace: $REFERENCE_ACTION"
echo ""

# Run the analysis
python3 /home/spirit-ai/pingpong_moz1/analyze_policy_performance.py \
    "$POLICY_TRACE" \
    --reference-obs "$REFERENCE_OBS" \
    --reference-action "$REFERENCE_ACTION" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Generating joint tracking analysis..."
python3 /home/spirit-ai/pingpong_moz1/analyze_joint_tracking.py \
    "$POLICY_TRACE" \
    --reference "$REFERENCE_OBS" \
    --output "$OUTPUT_DIR/joint_tracking_analysis.png"

echo ""
echo "Generating detailed observation comparison plots (67 dimensions)..."
python3 /home/spirit-ai/pingpong_moz1/plot_detailed_obs_comparison.py \
    "$POLICY_TRACE" \
    "$REFERENCE_OBS" \
    --output-dir "$OUTPUT_DIR/obscomparison"

echo ""
echo "========================================================================"
echo "Analysis complete!"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
