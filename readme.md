# 相机标定
激活conda环境
  conda activate pingpong

进入工作目录
  cd /home/spirit-ai/pingpong_moz1

确认ROS2环境已source
  source /opt/ros/humble/setup.bash  # 或你的ROS2版本
  source install/setup.bash

启动数据采集
ros2 launch realsense2_camera rs_launch.py \
    rgb_camera.color_profile:=848x480x60 \
    enable_color:=true \
    enable_depth:=false
    
  python3 src/pingpong_moz1/pingpong_controller/tools/vision_calib/collect_eye_to_hand.py

设置变量（替换为你的实际采集目录）
  SAMPLE_DIR="src/pingpong_moz1/pingpong_controller/data/vision_calib/eye_to_hand_samples_20260611_171730"

运行标定（使用去畸变模式）
  python3  src/pingpong_moz1/pingpong_controller/tools/vision_calib/eye_hand_calib.py \
    --intrinsics ${SAMPLE_DIR}/camera_intrinsics_raw.json \
    --samples ${SAMPLE_DIR}/samples.txt \
    --image-dir ${SAMPLE_DIR}/images \
    --tag-family tag36h11 \
    --tag-rows 6 \
    --tag-cols 6 \
    --tag-size 0.025 \
    --tag-spacing 0.3 \
    --method PARK \
    --undistort \
    --undistort-alpha 0.0 \
    --min-tags 8 \
    --max-reproj-error 10.0 \
    --corner-order 0123 \
    --debug-dir ${SAMPLE_DIR}/debug_vis



# 目标检测
启动相机驱动
ros2 launch realsense2_camera rs_launch.py \
    rgb_camera.color_profile:=848x480x60 \
    enable_color:=true \
    enable_depth:=false
目标检测代码
colcon build --packages-select pingpong_controller

source install/setup.bash 

ros2 run pingpong_controller pingpong_node --ros-args \
      -p enable_rl_policy:=false \
      -p enable_init_pose:=false \
      -p image_width:=848 \
      -p image_height:=480 \
      -p infer_imgsz:=480  \
      -p enable_visualization:=true \
      -p use_kalman:=true \
      -p enable_fov_loss_compensation:=true
启用visualization会降低检测频率。已不会，可视化单独开了一个线程。

保存原始视频再标注用于DEBUG，要先关闭相机驱动
ros2 run pingpong_controller pingpong_node --ros-args \
    -p use_direct_realsense:=true \
    -p enable_rl_policy:=false \
    -p enable_init_pose:=false \
    -p image_width:=848 \
    -p image_height:=480 \
    -p realsense_fps:=60 \
    -p infer_imgsz:=480 \
    -p enable_visualization:=true \
    -p use_kalman:=true \
    -p enable_fov_loss_compensation:=true \
    -p save_raw_video:=true \
    -p raw_video_fps:=60.0


python3 src/pingpong_moz1/pingpong_controller/tools/annotate_raw_video.py \
    -i src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_<时间戳>.mp4 \
    --infer-imgsz 480

## 动捕 UDP 输入
当前接收端支持 JSON 包：

```json
{"marker_frmIdx":123,"marker_fTimestamp":1.234,"marker_1_px":1000,"marker_1_py":2000,"marker_1_pz":1500}
```

`marker_1_px/py/pz` 会作为动捕坐标系下的球心位置；如果包里没有 `marker_1_vx/vy/vz`，节点会用相邻 UDP 帧自动估计速度。`mocap_position_unit:=auto` 会把大于 20 的坐标自动当 mm，否则当 m；如果现场确定单位，建议显式设成 `m` 或 `mm`。

动捕到机器人 `base_link` 的标定：

```bash
ros2 run pingpong_controller mocap_calibration -- \
    --udp-port 9870 \
    --marker-id 1 \
    --position-unit auto \
    --timestamp-unit s \
    --robot-xml src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml \
    --output mocap_calibration.txt
```

只验证动捕球状态，不启动视觉：

```bash
ros2 run pingpong_controller pingpong_node --ros-args \
    -p enable_mocap_input:=true \
    -p mocap_udp_port:=9870 \
    -p mocap_marker_id:=1 \
    -p mocap_position_unit:=auto \
    -p mocap_timestamp_unit:=s \
    -p mocap_calibration_path:=mocap_calibration.txt \
    -p enable_rl_policy:=false \
    -p enable_init_pose:=false \
    -p enable_ball_state_estimator:=true \
    -p publish_ball_state_debug:=true
```

如果 `marker_fTimestamp` 是毫秒/微秒/纳秒，分别把 `mocap_timestamp_unit` 改成 `ms`/`us`/`ns`。调试时可看 `/pingpong/ball_state`，轨迹文件仍写到 `outputs/pingpong_base_trajectory.*`。


## 200Hz 补偿 ball estimator
ros2 run pingpong_controller pingpong_node --ros-args \
      -p use_direct_realsense:=true \
      -p enable_rl_policy:=false \
      -p enable_init_pose:=false \
      -p image_width:=848 \
      -p image_height:=480 \
      -p realsense_fps:=60 \
      -p infer_imgsz:=480 \
      -p enable_visualization:=true \
      -p use_kalman:=false \
      -p enable_fov_loss_compensation:=false \
      -p save_raw_video:=true \
      -p raw_video_fps:=60.0 \
      -p enable_ball_state_estimator:=true \
      -p save_estimator_log:=true \
      -p ball_estimator_log_period_s:=0.5

## 椭圆检测+伪TrackerNet
ros2 run pingpong_controller pingpong_node --ros-args \
      -p use_direct_realsense:=true \
      -p enable_rl_policy:=false \
      -p enable_init_pose:=false \
      -p image_width:=848 \
      -p image_height:=480 \
      -p realsense_fps:=60 \
      -p infer_imgsz:=480 \
      -p enable_visualization:=true \
      -p use_kalman:=false \
      -p enable_fov_loss_compensation:=false \
      -p save_raw_video:=true \
      -p raw_video_fps:=60.0 \
      -p enable_ball_state_estimator:=true \
      -p save_estimator_log:=true \
      -p ball_estimator_log_period_s:=0.5 \
      -p enable_ellipse_fit:=true \
      -p enable_temporal_tracker:=true \
      -p tracker_max_miss_bridge:=2 \
      -p tracker_gate_px:=120.0 \

  enable_ellipse_fit          (默认 True)
  enable_temporal_tracker     (默认 False)
  tracker_max_miss_bridge     (默认 2)
  tracker_gate_px             (默认 120.0)
  一个诚实的提醒:方案1 是确定有益的(直接打到深度噪声这个根因),方案4 提升的是 2D 检测鲁棒性(快球/遮挡时框不掉)。但正如上轮分析,要把深度噪声从 ~80mm
  量级真正降到 10–20mm,仍需宽基线双目硬件——这两个软件改动是在现有相机上能做的最优,不是物理上限。

  建议下一步:开 enable_temporal_tracker:=true 跑一次,录 raw video 后用 annotate_raw_video.py --temporal-tracker
  离线对比,直观看门控/桥接是否让橙色框更跟手。要我帮你跑回放对比吗?

## 目标检测颜色收紧调整
python3 src/pingpong_moz1/pingpong_controller/tools/vision_calib/hsv_color_tuner.py     --width 848 --height 480 --fps 60     --yolo-model src/pingpong_moz1/pingpong_controller/models/best.pt     --infer-imgsz 480
按m切换视图，按s保存参数，按esc退出，按w保存json

## 室内光照条件下更鲁棒的启动方式：
ros2 run pingpong_controller pingpong_node --ros-args \
    -p use_direct_realsense:=true \
    -p enable_rl_policy:=false \
    -p enable_init_pose:=false \
    -p image_width:=848 \
    -p image_height:=480 \
    -p realsense_fps:=60 \
    -p infer_imgsz:=480 \
    -p enable_visualization:=true \
    -p use_kalman:=false \
    -p enable_fov_loss_compensation:=true \
    -p save_raw_video:=true \
    -p raw_video_fps:=60.0 \
    -p enable_ball_state_estimator:=true \
    -p save_estimator_log:=true \
    -p ball_estimator_log_period_s:=0.5 \
    -p hsv_lower:='[5,150,50]' \
    -p hsv_upper:='[30,255,255]' \
    -p color_match_threshold:=0.30 \
    -p color_circularity_min:=0.40

## 更鲁棒的离线标注
python3 src/pingpong_moz1/pingpong_controller/tools/annotate_raw_video.py \
      -i src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_<时间戳>.mp4 \
      --infer-imgsz 480 \
      --hsv-lower 5,150,50 \
      --hsv-upper 30,255,255 \
      --color-match-threshold 0.30 \
      --color-circularity-min 0.40 \
      --no-kalman \
      --calibration auto

## depth assist
ros2 run pingpong_controller pingpong_node --ros-args \
    -p use_direct_realsense:=true \
    -p enable_rl_policy:=false \
    -p enable_init_pose:=false \
    -p image_width:=848 \
    -p image_height:=480 \
    -p realsense_fps:=60 \
    -p infer_imgsz:=480 \
    -p enable_visualization:=true \
    -p use_kalman:=false \
    -p enable_fov_loss_compensation:=true \
    -p save_raw_video:=true \
    -p raw_video_fps:=60.0 \
    -p enable_ball_state_estimator:=true \
    -p save_estimator_log:=true \
    -p ball_estimator_log_period_s:=0.5 \
    -p hsv_lower:='[5,150,50]' \
    -p hsv_upper:='[30,255,255]' \
    -p color_match_threshold:=0.30 \
    -p color_circularity_min:=0.40 \
    -p use_depth_sensor:=true \
    -p depth_neighborhood_px:=2 \
    -p save_estimator_log:=true -p ball_estimator_log_period_s:=0.0

  # 硬件深度
  python3 src/pingpong_moz1/pingpong_controller/tools/annotate_raw_video.py \
    -i src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260630_201956.mp4 \
    -o src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260630_201956_hw_annotated.mp4 \
    --infer-imgsz 480 \
    --hsv-lower 5,150,50 \
    --hsv-upper 30,255,255 \
    --color-match-threshold 0.30 \
    --color-circularity-min 0.40 \
    --no-kalman \
    --calibration auto \
    --use-depth-sensor \
    --depth-scale 0.001 \
    --depth-neighborhood-px 2 \
    --detection-log src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260630_201956_hw_detections.csv


  # 单目对照
  python3 src/pingpong_moz1/pingpong_controller/tools/annotate_raw_video.py \
    -i src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260630_201956.mp4 \
    -o src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260630_201956_mono_annotated.mp4 \
    --infer-imgsz 480 \
    --hsv-lower 5,150,50 \
    --hsv-upper 30,255,255 \
    --color-match-threshold 0.30 \
    --color-circularity-min 0.40 \
    --no-kalman \
    --calibration auto \
    --detection-log src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260630_201956_mono_detections.csv



## rviz可视化球轨迹
source install/setup.bash
ros2 run pingpong_controller enhanced_trajectory_visualizer \
      --ros-args \
      -p csv_file:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/pingpong_base_trajectory.csv \
      -p time_start:=13.0 \
      -p time_end:=15.0 \
      -p time_range_mode:=relative \
      -p color_by_velocity:=true

可以指定开始时间和结束时间
  RViz设置：
  1. 启动 rviz2
  2. Add → By topic → /ball_trajectory
  3. Fixed Frame 设为 base_link


## 针对一个视频提取YOLO标注数据
python3 extract_frames_for_labeling.py \
      --video /home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260701_160142.mp4 \
      --output /home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/dataset/reflective_ball/images \
      --interval 1 \
      --max-frames 5000

## 红外检测参数调节
python ir_tuning_tool.py

## YOLO+红外反光混合检测方法测试
python3 src/pingpong_moz1/pingpong_controller/hybrid_ball_detector.py          --model src/pingpong_moz1/pingpong_controller/models/best.pt          --params ir_params.json          --yolo-interval 1          --yolo-conf 0.25           --print-every 10

## 混合检测方案
ros2 run pingpong_controller pingpong_node --ros-args       -p use_direct_realsense:=true       -p enable_rl_policy:=false       -p enable_init_pose:=false       -p image_width:=848       -p image_height:=480       -p realsense_fps:=60       -p infer_imgsz:=480       -p enable_visualization:=true       -p use_kalman:=false       -p enable_fov_loss_compensation:=true       -p save_raw_video:=true       -p raw_video_fps:=60.0       -p enable_ball_state_estimator:=true       -p save_estimator_log:=true       -p ball_estimator_log_period_s:=0.0       -p use_depth_sensor:=true       -p depth_neighborhood_px:=2       -p yolo_model_path:=src/pingpong_moz1/pingpong_controller/models/best.pt       -p use_hybrid_detector:=true       -p ir_params_json:=ir_params.json       -p target_class:='Class 0'       -p conf_threshold:=0.25 \
-p save_ir_frames:=true  -p save_depth_frames:=true 

## YOLO+红外反光混合检测方法离线视频标注
python  src/pingpong_moz1/pingpong_controller/tools/annotate_raw_video_hybrid.py     -i ~/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_2026xxxx_xxxxxx.mp4 --output-frame base_link --output ~/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_2026xxxx_xxxxxx_hybrid_annotatedv1.mp4


## 深度拟合模型，标定RGB和IR的深度残差模型
python3 yolo_depth_calibration.py src/pingpong_moz1/pingpong_controller/models/best.pt
离线拟合
python3 offline_motion_depth_calibrator.py     src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_20260702_175827.mp4     src/pingpong_moz1/pingpong_controller/models/best.pt


# 实验下发ros2 topic mx_mix_command
ros2 topic pub /mx_mix_command mc_core_interface/msg/MechUnitCmdArray "{header: {frame_id: ''}, cmds: [{mu_idx: 3, jnt_pos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], end_pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, psi: 0.0, use_jnt: true}]}"

## 实物仿真验证策略（录制球轨迹、仿真球轨迹）
ros2 run pingpong_controller pingpong_node --ros-args \
    -p enable_rl_policy:=true \
    -p policy_backend:=mjx \
    -p rl_model_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/inverse_compensation/18_stage4g_strong_contact_dr_inverse_mpc.pkl \
    -p enable_delay_conditioning:=true \
    -p robot_xml_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml \
    -p control_rate_hz:=200.0 \
    -p enable_replay_mode:=true \
    -p replay_csv_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/estimator_log_20260702_211743_replay_12s.csv \
    -p replay_loop:=false \
    -p enable_init_pose:=false \
    -p enable_ball_lost_failsafe:=false \
    -p enable_ball_state_estimator:=false \
    -p enable_safety_limiter:=false \
    -p publish_rl_joint_cmd_debug:=true \
    -p right_arm_mu_idx:=3 \
    -p -p params_file:=config_done_monitoring.yaml

建议直接运行：
/RUN_POLICY_ANALYSIS.sh 

# 实物replay仿真轨迹
colcon build --packages-select pingpong_replay

source install/setup.bash 

ros2 run pingpong_replay replay_node --ros-args     -p csv_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/stage4g_action_trace.csv     -p right_arm_mu_idx:=3   -p input_mode:=action

对比分析
ros2 run pingpong_replay plot_analysis /home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/replay_analysis_1781246614.csv -o /home/spirit-ai/pingpong_moz1/src/pingpong_replay/comparison.png --error-plot /home/spirit-ai/pingpong_moz1/src/pingpong_replay/error.png

## compensation replay
inverse smith & inverse mpc
ros2 run pingpong_replay replay_node \
    --ros-args \
    -p csv_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/inverse_compensation/validate_trace_inverse_mpc.csv \
    -p input_mode:=action \
    -p compensation_mode:=inverse_mpc \
    -p right_arm_mu_idx:=3

# 机械臂右臂工作空间标定
python tools/calibrate_workspace.py
  命令列表:
  - Enter        → 记录当前末端位置为边界点
  - current      → 显示当前末端位置 [x, y, z]
  - plot         → 打开3D可视化窗口（显示已记录的点+工作空间盒子）
  - track        → 开启/关闭轨迹跟踪（移动机械臂时自动记录轨迹）
  - done         → 完成标定，输出配置参数

  Add these lines to config_done_monitoring.yaml:
  ------------------------------------------------------------
      done_ee_x_limits_m: [-0.300, 0.300]
      done_ee_y_limits_m: [0.200, 0.800]
      done_ee_z_limits_m: [0.600, 1.400]
  ------------------------------------------------------------


# 模拟各种情况
## 模拟球轨迹生成
python3 src/pingpong_moz1/pingpong_controller/tools/generate_safety_replay_trajectories.py \
  --scenario all \
  --duration 6.0 \
  --rate-hz 60 \
  --frame sim \
  --output /tmp/pingpong_safety_replay

## 实物仿真replay仿真数据
./RUN_POLICY_ANALYSYS.sh

## 模拟球
./RUN_POLICY_SAFETY_SCENARIOS.sh
或
./RUN_POLICY_SAFETY_SCENARIOS.sh up_occlusion
或
./RUN_POLICY_SAFETY_SCENARIOS.sh stationary,jitter,lost_after_seen
或
./RUN_POLICY_SAFETY_SCENARIOS.sh up_occlusion -- \
  --space-z-max-m 1.75 \
  --throw-vz-m-s 4.0 \
  --invalid-fill continue

## ir_params.json参数调节
python3 __pycache__/ir_tuning_tool.cpython-310.pyc

# 实物执行策略

ros2 run pingpong_controller pingpong_node --ros-args \
  --params-file /home/spirit-ai/pingpong_moz1/config_done_monitoring.yaml \
  -p command_topic:=/mx_mix_command \
  -p joint_state_topic:=/joint_states \
  -p right_arm_mu_idx:=3 \
  -p use_direct_realsense:=true \
  -p use_hybrid_detector:=true \
  -p ir_params_json:=/home/spirit-ai/pingpong_moz1/ir_params.json \
  -p yolo_model_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/models/best.pt \
  -p calibration_json_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/vision_calib/eye_hand_result_848_undistorted.json \
  -p image_width:=848 \
  -p image_height:=480 \
  -p realsense_fps:=60 \
  -p target_class:='Class 0' \
  -p conf_threshold:=0.25 \
  -p color_match_threshold:=0.0 \
  -p enable_rl_policy:=true \
  -p policy_backend:=mjx \
  -p rl_model_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/rl_sim/inverse_compensation/18_stage4g_strong_contact_dr_inverse_mpc.pkl \
  -p enable_delay_conditioning:=true \
  -p robot_xml_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml \
  -p control_rate_hz:=200.0 \
  -p enable_init_pose:=true \
  -p require_enter_confirm:=true \
  -p post_enter_start_delay_s:=3.0 \
  -p run_duration_s:=20.0 \
  -p enable_safety_limiter:=false \
  -p enable_ball_lost_failsafe:=true \
  -p enable_ball_capture_gate:=true \
  -p enable_ball_state_estimator:=true \
  -p use_depth_sensor:=true \
  -p enable_visualization:=true \
  -p start_confirm_topic:=/pingpong/start_control \
  -p auto_continue_on_stdin_eof:=false \
  -p enter_prompt_log_period_s:=5.0 \
  -p quiet_vision_logs:=true \
  -p print_every:=0 \
  -p depth_correction_model_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/depth_correction_model.pkl

或运行
./RUN_REAL_POLICY_EXPERIMENT.sh

## 终极模拟
ENABLE_ROS_BRIDGE=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

ENABLE_CAMERA_BALL=true ENABLE_ROS_BRIDGE=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh