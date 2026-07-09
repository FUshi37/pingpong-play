#!/bin/bash
# Launch a local browser-based policy safety probe. The Python backend loads
# the MJX deployment policy and simulates the high-level safety gates.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/home/spirit-ai/pingpong_moz1}"
PROBE_SCRIPT="$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/tools/interactive_policy_safety_probe.py"
SETUP_FILE="${SETUP_FILE:-$WORKSPACE_DIR/install/setup.bash}"

MODEL_PATH="${MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/inverse_compensation/18_stage4g_strong_contact_dr_inverse_mpc.pkl}"
ROBOT_XML="${ROBOT_XML:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml}"
POLICY_OBS_DIM="${POLICY_OBS_DIM:-67}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

DONE_Z_MIN_M="${DONE_Z_MIN_M:-0.3}"
DONE_Z_MAX_M="${DONE_Z_MAX_M:-1.9}"
DONE_X_RANGE_M="${DONE_X_RANGE_M:-0.5}"
DONE_Y_RANGE_M="${DONE_Y_RANGE_M:-0.5}"
DONE_RACKET_ANCHOR_DISTANCE_M="${DONE_RACKET_ANCHOR_DISTANCE_M:-1.1}"
DONE_RACKET_Z_HIGH_OFFSET_M="${DONE_RACKET_Z_HIGH_OFFSET_M:-0.24}"
DONE_RACKET_Z_LOW_OFFSET_M="${DONE_RACKET_Z_LOW_OFFSET_M:--0.12}"
DONE_EE_X_LIMITS_M="${DONE_EE_X_LIMITS_M:--0.298,0.163}"
DONE_EE_Y_LIMITS_M="${DONE_EE_Y_LIMITS_M:--0.502,-0.290}"
DONE_EE_Z_LIMITS_M="${DONE_EE_Z_LIMITS_M:-0.692,1.462}"
BALL_LOST_FAILSAFE_TIMEOUT_S="${BALL_LOST_FAILSAFE_TIMEOUT_S:-0.6}"
ENABLE_BALL_CAPTURE_GATE="${ENABLE_BALL_CAPTURE_GATE:-true}"
ENABLE_BALL_RELEASE_GATE="${ENABLE_BALL_RELEASE_GATE:-true}"
ENABLE_BALL_LOST_FAILSAFE="${ENABLE_BALL_LOST_FAILSAFE:-true}"
ENABLE_DONE_MONITORING="${ENABLE_DONE_MONITORING:-true}"
BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S="${BALL_RELEASE_MIN_DOWNWARD_VELOCITY_M_S:-0.15}"
BALL_RELEASE_MIN_Z_DROP_M="${BALL_RELEASE_MIN_Z_DROP_M:-0.015}"
BALL_RELEASE_WINDOW_S="${BALL_RELEASE_WINDOW_S:-0.12}"
BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS="${BALL_RELEASE_REQUIRED_CONSECUTIVE_TICKS:-3}"
BALL_RELEASE_MAX_OBS_AGE_S="${BALL_RELEASE_MAX_OBS_AGE_S:-0.08}"
BALL_NOISE_STD_M="${BALL_NOISE_STD_M:-0.0}"
BALL_OBSERVATION_RATE_HZ="${BALL_OBSERVATION_RATE_HZ:-60.0}"
PROBE_STEP_RATE_HZ="${PROBE_STEP_RATE_HZ:-60.0}"
PROBE_DISPLAY_RATE_HZ="${PROBE_DISPLAY_RATE_HZ:-60.0}"
ENABLE_BALL_STATE_ESTIMATOR="${ENABLE_BALL_STATE_ESTIMATOR:-true}"
BALL_ESTIMATOR_ENABLE_PREDICTION="${BALL_ESTIMATOR_ENABLE_PREDICTION:-true}"
BALL_ESTIMATOR_ENABLE_AERO_DRAG="${BALL_ESTIMATOR_ENABLE_AERO_DRAG:-true}"
BALL_ESTIMATOR_MAX_EXTRAPOLATION_S="${BALL_ESTIMATOR_MAX_EXTRAPOLATION_S:-0.45}"
BALL_ESTIMATOR_TRUST_LOSS_S="${BALL_ESTIMATOR_TRUST_LOSS_S:-0.10}"
BALL_ESTIMATOR_DEGRADE_LOSS_S="${BALL_ESTIMATOR_DEGRADE_LOSS_S:-0.40}"
BALL_ESTIMATOR_INVALID_AFTER_S="${BALL_ESTIMATOR_INVALID_AFTER_S:-0.60}"
BALL_ESTIMATOR_PREDICTION_HORIZON_S="${BALL_ESTIMATOR_PREDICTION_HORIZON_S:-0.0}"
ENABLE_ROS_BRIDGE="${ENABLE_ROS_BRIDGE:-true}"
ROS_PUBLISH_COMMANDS="${ROS_PUBLISH_COMMANDS:-true}"
COMMAND_TOPIC="${COMMAND_TOPIC:-/mx_mix_command}"
JOINT_STATE_TOPIC="${JOINT_STATE_TOPIC:-/joint_states}"
CART_STATE_TOPIC="${CART_STATE_TOPIC:-/cart_states}"
RIGHT_ARM_MU_IDX="${RIGHT_ARM_MU_IDX:-3}"
ENABLE_CAMERA_BALL="${ENABLE_CAMERA_BALL:-true}"
CAMERA_SOURCE_MODE="${CAMERA_SOURCE_MODE:-direct_realsense}"
USE_HYBRID_DETECTOR="${USE_HYBRID_DETECTOR:-true}"
USE_DEPTH_SENSOR="${USE_DEPTH_SENSOR:-true}"
IMAGE_WIDTH="${IMAGE_WIDTH:-848}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-480}"
REALSENSE_FPS="${REALSENSE_FPS:-60}"
CAMERA_IMAGE_TOPIC="${CAMERA_IMAGE_TOPIC:-/camera/camera/color/image_raw/compressed}"
CAMERA_RAW_IMAGE_TOPIC="${CAMERA_RAW_IMAGE_TOPIC:-/camera/camera/color/image_raw}"
CAMERA_IMAGE_TRANSPORT="${CAMERA_IMAGE_TRANSPORT:-auto}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/camera/color/camera_info}"
CALIBRATION_JSON_PATH="${CALIBRATION_JSON_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/outputs/vision_calib/eye_hand_result_848_undistorted.json}"
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$WORKSPACE_DIR/src/pingpong_moz1/pingpong_controller/models/best.pt}"
TARGET_CLASS="${TARGET_CLASS:-Class 0}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.25}"
IR_PARAMS_JSON="${IR_PARAMS_JSON:-$WORKSPACE_DIR/ir_params.json}"
YOLO_INTERVAL="${YOLO_INTERVAL:-1}"
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
INFER_IMGSZ="${INFER_IMGSZ:-640}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-0}"
DETECTOR_HALF="${DETECTOR_HALF:-false}"
USE_KALMAN="${USE_KALMAN:-false}"
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
CAMERA_VISUALIZATION_RATE_HZ="${CAMERA_VISUALIZATION_RATE_HZ:-60.0}"
CAMERA_WEB_MAX_WIDTH="${CAMERA_WEB_MAX_WIDTH:-848}"
CAMERA_WEB_JPEG_QUALITY="${CAMERA_WEB_JPEG_QUALITY:-62}"
ENABLE_CAMERA_COLOR_FILTER="${ENABLE_CAMERA_COLOR_FILTER:-false}"
SIM_CARTESIAN_X_OFFSET_M="${SIM_CARTESIAN_X_OFFSET_M:-0.0}"
SIM_CARTESIAN_Y_OFFSET_M="${SIM_CARTESIAN_Y_OFFSET_M:-0.023583}"
SIM_BASE_HEIGHT_M="${SIM_BASE_HEIGHT_M:-0.947396}"

ROS_BRIDGE_ARGS=()
CAMERA_ARGS=()
FLOW_ARGS=()
if [ "$ENABLE_BALL_CAPTURE_GATE" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-capture-gate)
fi
if [ "$ENABLE_BALL_RELEASE_GATE" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-release-gate)
fi
if [ "$ENABLE_BALL_LOST_FAILSAFE" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-lost-failsafe)
fi
if [ "$ENABLE_DONE_MONITORING" != "true" ]; then
    FLOW_ARGS+=(--disable-done-monitoring)
fi
if [ "$ENABLE_BALL_STATE_ESTIMATOR" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-state-estimator)
fi
if [ "$BALL_ESTIMATOR_ENABLE_PREDICTION" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-estimator-prediction)
fi
if [ "$BALL_ESTIMATOR_ENABLE_AERO_DRAG" != "true" ]; then
    FLOW_ARGS+=(--disable-ball-estimator-aero-drag)
fi
if [ "$ENABLE_ROS_BRIDGE" = "true" ] || [ "$ENABLE_CAMERA_BALL" = "true" ]; then
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
if [ "$USE_HYBRID_DETECTOR" != "true" ]; then
    CAMERA_ARGS+=(--disable-hybrid-detector)
fi
if [ "$USE_DEPTH_SENSOR" != "true" ]; then
    CAMERA_ARGS+=(--disable-depth-sensor)
fi
if [ "$DETECTOR_HALF" = "true" ]; then
    CAMERA_ARGS+=(--detector-half)
fi
if [ "$USE_KALMAN" = "true" ]; then
    CAMERA_ARGS+=(--use-kalman)
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

python3 "$PROBE_SCRIPT" \
    --host "$HOST" \
    --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --robot-xml-path "$ROBOT_XML" \
    --policy-obs-dim "$POLICY_OBS_DIM" \
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
    --done-racket-anchor-distance-m "$DONE_RACKET_ANCHOR_DISTANCE_M" \
    --done-racket-z-high-offset-m "$DONE_RACKET_Z_HIGH_OFFSET_M" \
    --done-racket-z-low-offset-m "$DONE_RACKET_Z_LOW_OFFSET_M" \
    --ee-x-limits-m="$DONE_EE_X_LIMITS_M" \
    --ee-y-limits-m="$DONE_EE_Y_LIMITS_M" \
    --ee-z-limits-m="$DONE_EE_Z_LIMITS_M" \
    --ball-noise-std-m "$BALL_NOISE_STD_M" \
    --ball-observation-rate-hz "$BALL_OBSERVATION_RATE_HZ" \
    --browser-step-rate-hz "$PROBE_STEP_RATE_HZ" \
    --display-rate-hz "$PROBE_DISPLAY_RATE_HZ" \
    --ball-estimator-max-extrapolation-s "$BALL_ESTIMATOR_MAX_EXTRAPOLATION_S" \
    --ball-estimator-trust-loss-s "$BALL_ESTIMATOR_TRUST_LOSS_S" \
    --ball-estimator-degrade-loss-s "$BALL_ESTIMATOR_DEGRADE_LOSS_S" \
    --ball-estimator-invalid-after-s "$BALL_ESTIMATOR_INVALID_AFTER_S" \
    --ball-estimator-prediction-horizon-s "$BALL_ESTIMATOR_PREDICTION_HORIZON_S" \
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
    "$@"
