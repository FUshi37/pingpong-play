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
动捕软件里把发送目标设置成机器人本机 IP 和端口，例如机器人 IP 是 `10.60.133.186`，端口用 `9870`。机器人端默认绑定 `0.0.0.0:9870`，也可以用 `mocap_bind_host` 指定某张网卡 IP。

当前接收端支持你之前给过的 JSON 包：

```json
{"marker_frmIdx":123,"marker_fTimestamp":1.234,"marker_1_px":1000,"marker_1_py":2000,"marker_1_pz":1500}
```

`marker_1_px/py/pz` 会作为动捕坐标系下的球心位置；如果包里没有 `marker_1_vx/vy/vz`，节点会用相邻 UDP 帧自动估计速度。接收端也兼容 `x/y/z`、`position:[x,y,z]`、`markers:[{"id":1,"position":[...]}]` 这几类常见 JSON 形态，以及老的二进制 `<d6f>` 包。`mocap_position_unit:=auto` 会把大于 20 的坐标自动当 mm，否则当 m；如果现场确定单位，建议显式设成 `m` 或 `mm`。

先只验证 UDP 链路和发布的球状态：

```bash
ros2 run pingpong_controller mocap_ball_state_node --ros-args \
    -p mocap_udp_port:=9870 \
    -p mocap_marker_id:=1 \
    -p mocap_position_unit:=auto \
    -p mocap_timestamp_unit:=s \
    -p mocap_calibration_path:=mocap_calibration.txt \
    -p ball_state_topic:=/pingpong/ball_state
```

另一端看输出：

```bash
ros2 topic echo /pingpong/ball_state
```

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

接入 `pingpong_node`，不启动视觉检测：

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

### legacy / simcode2 / simcode3 / simcode4 录制轨迹

新轨迹带有域随机化后的 action 缩放，并可能触发速度或关节限幅。两套轨迹
均采用“策略积分 → compensation → 执行器模型 → MuJoCo 动力学”的流程。
如果回放目标是复现仿真机械臂真正到达的运动轨迹，推荐直接将 CSV 的实际
仿真关节角 `arm_q` 作为实机参考，即使用 `input_mode:=sim_actual`：

```bash
# legacy: src/.../outputs/rl_sim/stage4g_action_trace.csv
ros2 run pingpong_replay replay_node --ros-args \
    -p trajectory_preset:=legacy \
    -p input_mode:=sim_actual \
    -p compensation_mode:=none \
    -p right_arm_mu_idx:=3

# simcode2: launch19_best2040_video_gpu1_seed20260722/action_trace.csv
ros2 run pingpong_replay replay_node --ros-args \
    -p trajectory_preset:=simcode2 \
    -p input_mode:=sim_actual \
    -p compensation_mode:=none \
    -p right_arm_mu_idx:=3

# simcode3: launch19_best2040_video_gpu1_seed20260722/action_trace.csv
ros2 run pingpong_replay replay_node --ros-args \
    -p trajectory_preset:=simcode3 \
    -p input_mode:=sim_actual \
    -p right_arm_mu_idx:=3

# simcode4: servo_submission_20260729/video/action_trace.csv
ros2 run pingpong_replay replay_node --ros-args \
    -p trajectory_preset:=simcode4 \
    -p input_mode:=sim_actual \
    -p right_arm_mu_idx:=3
```

`sim_actual` 精确读取 `arm_q`，对比分析中的 theoretical position/velocity/
acceleration 也分别读取 `arm_q`、`arm_qvel`、`arm_qacc`。必须保持
`compensation_mode:=none`（默认值），否则会改变这条已经执行完成的轨迹，节点
会拒绝启动。

完整回放会先移动到 CSV 第一个仿真步之前的 reset 姿态，再开始发送 `arm_q`。
legacy 使用自己的 reset 姿态；simcode2、simcode3、simcode4 使用相同的
warm-start reset 姿态。设置 `start_index>0` 时则直接移动到裁剪后轨迹首点。

其他模式仍可用于分层诊断：`recorded` 读取补偿前的 `arm_cmd_q`；
`compensated` 读取补偿及安全限幅后的 `arm_safe_q`；`arm_applied_q` 是送给
MuJoCo PD 的目标角，而不是仿真关节真正到达的角度。

所有回放分析数据默认统一保存到：

```text
~/pingpong_moz1/src/pingpong_replay/pingpong_replay/outputs
```

同一次回放使用“preset + 时间戳”命名，例如 simcode3：

- `simcode3_<时间戳>_replay_analysis.csv`
- `simcode3_<时间戳>_replay_analysis.json`
- `simcode3_<时间戳>_comparison.png`
- `simcode3_<时间戳>_error.png`

可用 `auto_plot_analysis:=false` 关闭，或用 `comparison_plot_path`、
`error_plot_path` 指定文件名；用 `output_dir` 可以覆盖统一输出目录。原来的
`plot_analysis` 独立命令仍然可用。

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


BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=0.05 ENABLE_CAMERA_BALL=true ENABLE_ROS_BRIDGE=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh


按当前 checkpoint 的 200Hz policy / 50Hz 球观测契约运行：每个输出帧只做一次延迟补偿预测，随后保持到下一帧：

BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=predict_on_frame_hold \
BALL_ESTIMATOR_FRAME_HOLD_OUTPUT_RATE_HZ=50.0 \
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true \
ENABLE_CAMERA_BALL=true ENABLE_ROS_BRIDGE=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

`BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true` 会在 release gate 清除前使用无重力的常速度模型，清除后根据近期真实测量重置速度并切换回弹道模型。原连续预测通路仍保留，设置 `BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=continuous` 即可恢复。关闭 `BALL_ESTIMATOR_ENABLE_PREDICTION` 仍是只保持最新滤波状态且完全不外推的第三条通路。


BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=-1 \
RL_USE_VALID_BALL_OBS_AGE=false \
BALL_RELEASE_USE_ESTIMATOR_STATE=true \
ENABLE_CAMERA_BALL=true ENABLE_ROS_BRIDGE=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh


BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=predict_on_frame_hold \
BALL_ESTIMATOR_FRAME_HOLD_OUTPUT_RATE_HZ=50.0 \
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true \
BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=0.08 \
RL_USE_VALID_BALL_OBS_AGE=true \
BALL_RELEASE_USE_ESTIMATOR_STATE=false \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh


BALL_ESTIMATOR_R_LATERAL_MM=25 \
BALL_ESTIMATOR_R_DEPTH_MM=90 \
BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=predict_on_frame_hold \
BALL_ESTIMATOR_FRAME_HOLD_OUTPUT_RATE_HZ=60.0 \
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true \
BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=0.08 \
RL_USE_VALID_BALL_OBS_AGE=true \
BALL_RELEASE_USE_ESTIMATOR_STATE=false \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh


BALL_ESTIMATOR_R_LATERAL_MM=25.0 \
BALL_ESTIMATOR_R_DEPTH_MM=90.0 \
PROBE_CONTROL_SUBPROCESS=true \
PROBE_BACKEND_CONTROL_LOOP=true \
CONTROL_RATE_HZ=200.0 \
REALSENSE_FPS=60 \
BALL_OBSERVATION_RATE_HZ=60.0 \
ENABLE_CAMERA_BALL=true \
CAMERA_SOURCE_MODE=direct_realsense \
USE_HYBRID_DETECTOR=true \
USE_DEPTH_SENSOR=true \
ENABLE_CAMERA_DETECTION=true \
BALL_ESTIMATOR_ENABLE_PREDICTION=true \
BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=continuous \
BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=0.08 \
RL_USE_VALID_BALL_OBS_AGE=true \
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true \
BALL_RELEASE_USE_ESTIMATOR_STATE=false \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh


CONTROL_RATE_HZ=200.0 \
ENABLE_SAFETY_LIMITER=true \
BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=continuous \
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true \
BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=0.08 \
RL_USE_VALID_BALL_OBS_AGE=true \
BALL_RELEASE_USE_ESTIMATOR_STATE=false \
./RUN_REAL_POLICY_EXPERIMENT.sh -- \
  -p ball_estimator_r_lateral_mm:=25.0 \
  -p ball_estimator_r_depth_mm:=90.0



log

BALL_ESTIMATOR_R_LATERAL_MM=25.0 \
BALL_ESTIMATOR_R_DEPTH_MM=90.0 \
PROBE_CONTROL_SUBPROCESS=true \
PROBE_BACKEND_CONTROL_LOOP=true \
CONTROL_RATE_HZ=200.0 \
REALSENSE_FPS=60 \
BALL_OBSERVATION_RATE_HZ=60.0 \
ENABLE_CAMERA_BALL=true \
CAMERA_SOURCE_MODE=direct_realsense \
USE_HYBRID_DETECTOR=true \
USE_DEPTH_SENSOR=true \
ENABLE_CAMERA_DETECTION=true \
BALL_ESTIMATOR_ENABLE_PREDICTION=true \
BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=continuous \
BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=0.08 \
RL_USE_VALID_BALL_OBS_AGE=true \
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true \
BALL_RELEASE_USE_ESTIMATOR_STATE=false \
PROBE_RECORDING_CONTROL_RATE_HZ=200.0 \
PROBE_RECORDING_IMAGE_RATE_HZ=60.0 \
PROBE_RECORDING_ANALYSIS_VIDEO_FPS=60.0 \
PROBE_RECORDING_QUEUE_SIZE=1024 \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh


# 我觉得只通过末端限制还是不太够，当末端到达时加速度和角速度并不是0导致机械臂还会只需运动，因此我觉得需要从加速度和角速度入手计算预测位置，根据预测位置提前终止。
BALL_ESTIMATOR_R_LATERAL_MM=25.0 \
BALL_ESTIMATOR_R_DEPTH_MM=90.0 \
PROBE_CONTROL_SUBPROCESS=true \
PROBE_BACKEND_CONTROL_LOOP=true \
CONTROL_RATE_HZ=200.0 \
REALSENSE_FPS=60 \
BALL_OBSERVATION_RATE_HZ=60.0 \
ENABLE_CAMERA_BALL=true \
CAMERA_SOURCE_MODE=direct_realsense \
USE_HYBRID_DETECTOR=true \
USE_DEPTH_SENSOR=true \
ENABLE_CAMERA_DETECTION=true \
BALL_ESTIMATOR_ENABLE_PREDICTION=true \
BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=continuous \
BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=0.08 \
RL_USE_VALID_BALL_OBS_AGE=true \
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true \
BALL_RELEASE_USE_ESTIMATOR_STATE=false \
PROBE_RECORDING_CONTROL_RATE_HZ=200.0 \
PROBE_RECORDING_IMAGE_RATE_HZ=60.0 \
PROBE_RECORDING_ANALYSIS_VIDEO_FPS=60.0 \
PROBE_RECORDING_QUEUE_SIZE=1024 \
WORKSPACE_SAFETY_MODE=predictive \
ENABLE_SIM_MIRROR=prompt \
SIM_FRAME_RATE_HZ=60.0 \
SIM_RENDER_WIDTH=640 \
SIM_RENDER_HEIGHT=360 \
SIM_USE_MUJOCO_RENDERER=false \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

BALL_ESTIMATOR_R_LATERAL_MM=25.0 \
BALL_ESTIMATOR_R_DEPTH_MM=90.0 \
PROBE_CONTROL_SUBPROCESS=true \
PROBE_BACKEND_CONTROL_LOOP=true \
CONTROL_RATE_HZ=200.0 \
REALSENSE_FPS=60 \
BALL_OBSERVATION_RATE_HZ=60.0 \
ENABLE_CAMERA_BALL=true \
CAMERA_SOURCE_MODE=direct_realsense \
USE_HYBRID_DETECTOR=true \
USE_DEPTH_SENSOR=true \
ENABLE_CAMERA_DETECTION=true \
BALL_ESTIMATOR_ENABLE_PREDICTION=true \
BALL_ESTIMATOR_CONTROL_OUTPUT_MODE=continuous \
BALL_ESTIMATOR_CONTROL_MAX_EXTRAPOLATION_S=0.08 \
RL_USE_VALID_BALL_OBS_AGE=true \
BALL_ESTIMATOR_RELEASE_GATED_DYNAMICS=true \
BALL_RELEASE_USE_ESTIMATOR_STATE=false \
PROBE_RECORDING_CONTROL_RATE_HZ=200.0 \
PROBE_RECORDING_IMAGE_RATE_HZ=60.0 \
PROBE_RECORDING_ANALYSIS_VIDEO_FPS=60.0 \
PROBE_RECORDING_QUEUE_SIZE=1024 \
WORKSPACE_SAFETY_MODE=predictive \
ENABLE_SIM_MIRROR=true \
SIM_MIRROR_DEFAULT_ENABLED=true \
SIM_BACKEND=mjx \
SIM_FRAME_RATE_HZ=60.0 \
SIM_RENDER_WIDTH=640 \
SIM_RENDER_HEIGHT=360 \
SIM_USE_MUJOCO_RENDERER=true \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

# mujoco仿真中的流程不应该是action->执行器模型->compensation吗？参考simcode，这才是正确的流程。执行器模型是对实物机器人关节特性的表征，快速响应的PD+执行器模型能够比较好的表现出实物上执行延迟+跟踪误差的特性。以及安全机制感觉还有BUG，不够完美（阻抗控制？）。


# 还是不对，机械臂还是会大幅度移动，我想要的效果是机器人在边界时有个把机械臂向回拉的力，而不是把机器人停住然后剧烈移动，我觉得是现在的方法有问题，实现predictive模式之前的限制是我想要的比较合理的效果，在碰到边界时会在边界徘徊。是不是用阻抗控制的形式实现更合理？

# 当前 position_soft 与 position hard 使用相同边界和正常命令路径。末端任意
# x/y/z 方向越界（或出现其他 done）后立即终止策略，并持续按 reset 限速发送
# 当前 RESET_INITIAL_POSE。实机发布时边界由关节反馈 FK 判定，不再依赖命令 FK。
# 启动脚本默认读取仓库根目录的 config_done_monitoring.yaml。优先级为：
# 显式环境变量 > YAML > 脚本默认值。可先只打印最终配置而不启动探针：
PROBE_CONFIG_ONLY=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

WORKSPACE_SAFETY_MODE=position_soft \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

# 使用其他配置文件，或明确禁用 YAML 加载：
CONFIG_DONE_MONITORING_YAML=/path/to/config.yaml ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
LOAD_CONFIG_DONE_MONITORING=false ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

# 实验性的连续虚拟弹簧/阻尼仍可单独选择，不再是 position_soft 的实现。
WORKSPACE_SAFETY_MODE=impedance_soft \
WORKSPACE_SOFT_MARGIN_M=0.05 \
WORKSPACE_SOFT_PULL_SPEED_M_S=0.08 \
WORKSPACE_SOFT_DAMPING=1.0 \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh


# 还是不行，我只是想让机械臂在末端在安全边界附近移动，能不能当机械臂末端到达边界时机械臂持续输出触发安全限制之前一刻的指令，当回到安全范围内时就恢复正常的指令发送。这样是不是能起到在安全边界附近也不会大幅度移动的效果。

# 我感觉现在interactive_policy_safety_probe.py代码太长了，能不能按照功能分块，分成数个代码文件，这样可读性更强一些

# 然后三个soft软约束依然没到达我想要的效果，我期望机械臂末端触碰到边界时实现的效果是在边界点振荡，如果球移动导致策略控制机械臂移动，那么机械臂也能在移动方向上移动（如果移动方向没有超出边界），同时触碰边界的方向上如果还在边界上持续振荡。这个振荡效果应该是策略控制机械臂末端向外移动，同时安全软约束控制机械臂末端不向外移动共动导致的。这里需要注意compensation机制，compensation的影响似乎比较大。

我觉得颠乒乓球并不是一个特别难实现的惹怒，利用mpc应该也可以实现。我已经能通过相机获取球的状态了，而且xml文件里也有乒乓球拍的body，可以计算乒乓球拍中心的位置，那么颠球应该并不难实现。按步骤实现以下功能，先只实现第一个，实现完成后暂停目标，我会自己进行实验测试：
1. 乒乓球拍中心跟踪乒乓球，这一阶段实现后我会手持球移动，机器人需要将乒乓球中心跟踪在乒乓球下方，追踪高度差高度可以自由设定（这个可用于检测球状态检测是否准确）；
2. 实现颠球功能，根据乒乓球预测其运动轨迹实现颠球功能，根据乒乓球预测其运动轨迹并颠球。

## 阶段 1：球拍中心跟踪手持球

本阶段只实现位置跟踪，用于手持球缓慢移动时检查相机、标定和球状态输出是否可信；不包含抛球、落点预测、击球时序或连续颠球功能。阶段 2 现在作为独立的 `mpc_juggle` 模式提供，启动按钮仍默认关闭。

### 跟踪目标与坐标系

控制目标定义为：

```text
racket_target = control_ball - [0, 0, height_offset_m]
```

也就是球拍中心与球心在 x/y 方向对齐，球拍在球下方保持 `height_offset_m` 的竖直距离。例如高度差设为 `0.25m` 时，目标球拍高度为 `ball_z - 0.25m`。

XML 中用于 FK、Jacobian 和误差计算的球拍中心是 `right_ee_site`。它位于拍面中心，不是 `right_racket` body 的原点；网页显示的 racket/target/error 也应按这个 site 解释。

跟踪只接受以下已标定坐标输入：

- 检测输出为 `base_link`，随后按部署约定转换到控制侧 `sim_frame`；或
- 检测器已经直接输出 `sim_frame`。

若标定缺失、分辨率不匹配，或检测输出仍是 `camera`/未知 frame，跟踪控制会拒绝该球状态并保持当前位置，不允许直接拿相机坐标驱动机械臂。实机前应在网页确认 detector frame 为 `base_link`、control frame 为 `sim_frame`。

### 启动参数

脚本默认仍使用原策略模式：

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `DEFAULT_CONTROL_MODE` | `policy` | 网页初始控制模式；阶段 1 可设为 `racket_tracking` |
| `RACKET_TRACKING_HEIGHT_OFFSET_M` | `0.25` | 球心到拍面中心的竖直高度差，单位 m |
| `RACKET_TRACKING_USE_MPC` | `false` | 是否默认启用跟踪 MPC；网页仍可切换 |
| `RACKET_TRACKING_MAX_BALL_AGE_S` | `0.12` | 可用于跟踪的最大球状态年龄，单位 s |
| `RACKET_TRACKING_POSITION_GAIN_S` | `4.0` | 非 MPC 跟踪的位置误差增益 |
| `RACKET_TRACKING_DLS_DAMPING` | `0.04` | Jacobian DLS 阻尼 |
| `RACKET_TRACKING_MAX_CARTESIAN_SPEED_M_S` | `0.65` | 两种球拍 MPC 共用的最大笛卡尔速度，单位 m/s；只做阶段 1 时建议在网页降回 `0.20` |
| `RACKET_TRACKING_MAX_CARTESIAN_ACCELERATION_M_S2` | `16.0` | 两种球拍 MPC 共用的最大笛卡尔加速度，单位 m/s² |
| `RACKET_TRACKING_JOINT_VELOCITY_SCALE` | `0.25` | 球拍控制关节速度相对硬件上限的比例 |
| `RACKET_TRACKING_JOINT_ACCELERATION_SCALE` | `0.35` | 球拍控制关节加速度相对硬件上限的比例 |
| `RACKET_TRACKING_POSTURE_GAIN_S` | `0.12` | 冗余关节的参考姿态增益 |
| `RACKET_TRACKING_TARGET_BOUNDARY_MARGIN_M` | `0.005` | 目标相对工作空间边界的内缩量，单位 m |

其中速度与加速度默认值按手持球近距离初测做了保守限制；确认方向、标定和 hold 行为正确后，再逐项提高。

跟踪 MPC 的高级参数如下，只有启用 `RACKET_TRACKING_USE_MPC=true` 或网页 `Tracking MPC` 时才参与求解：

| 环境变量 | 默认值 |
| --- | --- |
| `RACKET_TRACKING_MPC_HORIZON_STEPS` | `24` |
| `RACKET_TRACKING_MPC_POSITION_WEIGHT` | `1.0` |
| `RACKET_TRACKING_MPC_TARGET_VELOCITY_WEIGHT` | `1.0` |
| `RACKET_TRACKING_MPC_VELOCITY_WEIGHT` | `0.0001`（历史名称，现作为 effort 权重） |
| `RACKET_TRACKING_MPC_SMOOTH_WEIGHT` | `0.0002` |
| `RACKET_TRACKING_MPC_POSTURE_WEIGHT` | `0.0001` |

先检查最终配置但不启动探针：

```bash
PROBE_CONFIG_ONLY=true \
DEFAULT_CONTROL_MODE=racket_tracking \
RACKET_TRACKING_HEIGHT_OFFSET_M=0.25 \
RACKET_TRACKING_USE_MPC=false \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

阶段 1 建议启动命令：

```bash
DEFAULT_CONTROL_MODE=racket_tracking \
RACKET_TRACKING_HEIGHT_OFFSET_M=0.25 \
RACKET_TRACKING_USE_MPC=false \
RACKET_TRACKING_MAX_BALL_AGE_S=0.12 \
BALL_ESTIMATOR_ENABLE_PREDICTION=false \
ENABLE_COMPENSATION=false \
ENABLE_CAMERA_BALL=true \
ENABLE_ROS_BRIDGE=true \
ROS_PUBLISH_COMMANDS=true \
WORKSPACE_SAFETY_MODE=position_soft \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

`ROS_PUBLISH_COMMANDS=true` 只允许网页下发命令，不会自动驱动机器人。网页里的 `Send command` 默认关闭，必须由操作者在检查完成后显式开启。

### 网页操作顺序

1. 打开探针网页，选择 `ball source = camera` 和 `control mode = Racket tracking`。
2. 设置 `height offset`；首次测试建议从 `0.25m` 开始，并确认目标仍在已标定工作空间内。
3. 按需切换 `Tracking MPC`。它控制球拍目标跟踪算法，与 `compensation` 下的 actuator `inverse_mpc` 是两层独立开关：前者计算如何跟随笛卡尔目标，后者补偿命令传输延迟和执行器滞后。首次低速验证建议保持两者都关闭；之后可以分别开关，不能把现有 `inverse_mpc` 当成跟踪 MPC。
4. 保持 `Send command` 关闭进行 dry-run。缓慢手持移动球，确认 camera/control ball、racket target、`right_ee_site` FK、tracking error 和状态均合理，且 detector/control frame 正确。
5. 检查工作空间边界、关节反馈新鲜度、急停和机器人周边环境。确认无误后再开启 `Send command`；开启后的第一条命令应先保持当前实测关节位置，然后才开始跟踪。
6. 先沿单一 x、y、z 方向小幅慢速移动球，再逐步组合移动。取消 `Send command` 时机器人应发送一次当前位置保持命令并停止连续下发。

### 丢球与安全行为

- 球状态无效、frame 不受支持、球状态年龄超过 `RACKET_TRACKING_MAX_BALL_AGE_S`，或关节反馈陈旧时，控制器必须进入 hold，不继续追逐旧位置或外推位置。
- 目标不可达或接近边界时，命令仍需经过工作空间终止判断、关节速度/加速度限制和命令心跳租约；不得绕过安全发布链路。
- 若机械臂已经到达硬关节边界，且“本周期不越界”和关节加速度限制在数学上无法同时满足，控制器优先保证关节位置与最终笛卡尔速度不越限，并在遥测中置 `accelerationConstraintOverridden=true`。出现该标志应停止实验、降低速度并检查工作空间，而不是继续依赖该覆盖行为。
- 发生 hard workspace stop 后，需要排查原因并执行 `Reset`，不能靠重新勾选 `Send command` 直接恢复。
- soft workspace 模式使用与 hard 相同的边界判定和正常命令路径。实机发布时，每个控制周期会在 capture/release gate 和既有 done 锁存之前，用关节反馈 FK 检查末端 x/y/z；任意方向越界或出现其他终止条件后都会终止策略，并由专用 reset 通道按限速持续发送当前所选 `RESET_INITIAL_POSE`，直到操作者执行 `Reset` 清除锁存，不再进入旧 parking/CBF 路径。网页和录制中的 `workspaceSafety.poseSource` 应为 `joint_state_fk`；`stale_joint_state_fk` 或 `command_fk_fallback_no_joint_state` 表示关节反馈需要排查。
- 手持球验证默认建议关闭弹道预测，避免静止或缓慢移动的球被重力模型向下外推。

### 阶段 1 验收步骤

1. **Dry-run**：`Send command` 关闭时移动球，机器人不应收到连续控制命令；网页目标应满足 `target_x=ball_x`、`target_y=ball_y`、`target_z=ball_z-height_offset`。
2. **静止跟踪**：将球保持在安全可达位置，拍面中心稳定在球下方；记录横向误差和高度差误差，建议先以 P95 不超过 `0.05m` 作为初始实验门槛，再根据视觉噪声和机械臂响应收紧。
3. **分轴移动**：分别沿 x/y/z 缓慢移动，球拍应跟随可达方向；soft 模式触及工作空间限制时应终止跟踪，并按 reset 限速平滑回到所选初始姿态，不应出现大幅跳动或继续向外运动。
4. **丢球测试**：遮挡球或停止有效检测超过 `0.12s`，页面应显示 stale/lost/hold，球拍停止追踪；恢复检测后应平滑恢复，不能发生关节目标突跳。若关闭 `Send command` 时关节反馈也已超过 50 ms，后端会直接关闭命令流而不再发布一个由旧反馈构造的“保持”目标。
5. **MPC 对照**：保持其他参数不变，分别关闭和开启 `Tracking MPC` 录制同一段手持轨迹；两种模式都必须遵守相同安全限制，并比较 tracking error、命令平滑度和饱和次数。
6. **停止测试**：取消 `Send command`，确认只发送一次实测位置 hold；关闭页面或中断网页心跳后，也应在命令租约超时后保持并停止连续下发。

上述验收通过后再进入阶段 2；不要跳过相机 frame、球状态年龄、保持与停止测试。

## 阶段 2：MPC 自起球与连续颠球

网页控制模式 `MPC juggling` 使用用户给定的右臂放球姿态：

```text
degrees = [5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214]
radians = [0.100112086, -0.774908735, 0.535519374, 1.695447742,
           0.860848747, -0.214134446, 0.248081100]
```

这个姿态下 `right_ee_site` 的拍面法向接近控制坐标系 `+Z`。选择模式后，低层控制器先按共享关节速度/加速度限制移动到该姿态；这一步不要求已经看到球，也不会自动启动起球。

### 控制结构

阶段 2 不是把阶段 1 的球位置直接当作击球目标。它包含两层滚动控制：

1. 外层 hybrid first-exit MPC 每个控制 tick 只预测到下一次下降接触。它不再假设球必定穿过一个固定高度，而是沿预测弹道搜索 `racket = ball - radius * normal` 的移动接触事件，并同时检查该事件的位置、碰撞速度、共享速度/加速度上限和工作空间是否可达。事件末端的位置与速度是 MPC 硬约束；全部离散 knot 与每段内部的位置极值都必须留在低层 target margin 收缩后的工作空间内。整段笛卡尔加速度只执行第一步，然后用最新相机状态重算。无法在限幅内满足碰撞边界时会 hold，不会把有明显终端误差的软解当作有效击球。
2. 内层 Jacobian MPC 把外层给出的下一拍位置/速度参考转换为 7 轴位置命令，并继续经过已有的关节位置、速度、加速度、工作空间、命令租约和 ROS 发布安全链。

目标反弹竖直速度由目标 apex 高度得到：

```text
v_ball_out_z = sqrt(2 * g * apex_height)
v_racket_z = (v_ball_out_z + restitution * v_ball_in_z)
             / (1 + restitution)
```

横向回中依赖 `tangential transfer` 的近似模型。首次低 apex 反弹会保留入射切向速度，只控制法向反弹；成功反弹并逐步提高 apex 后，控制器会在 4 拍内逐渐请求完整横向回中，实际比例仍会按共享速度/加速度上限自动缩小，避免给约 2° 倾斜的初始拍面提出不可达的横向加速度。这个系数会随胶皮、入射速度和旋转改变，必须用实机数据标定；若设得不准，平拍平移控制能追到落点，但不能保证长期消除水平漂移。后续可在现有旋转 Jacobian 上增加小倾角闭环。

### 启动状态机

启动按钮是边沿触发且默认 `OFF`。后端状态按下列顺序运行：

```text
positioning -> ready_wait -> countdown -> launch(lift + downstroke)
                                      -> flight MPC -> contact_wait -> flight MPC
                                              \-----------> fault_hold
```

- `ready_wait` 同时检查初始关节角、拍面朝上、球心到拍面约一个球半径、横向位于拍面中心容差内、球/球拍速度足够小，以及球状态未过期；条件需稳定保持一段时间。
- 倒计时默认 3 秒，由后端固定步控制时钟计算。MPC 事件网格也使用同一时钟，避免 ROS/Python 调度在 5 ms 周期附近抖动时跳过末端 knot；墙上单调时钟仍单独负责命令租约和反馈年龄等真实超时安全检查。控制循环过载时倒计时只会更保守地变长。倒计时期间球被移走或状态变旧会返回等待，不会继续起球。
- 自起球使用光滑的 cosine lift-and-release。拍先向上带球，再产生大于重力的向下加速度使单边接触打开；离拍后仍继续一段下撤轨迹，避免在拍面最高点立刻切换 MPC 而过早二次碰球。控制器会预先检查峰值速度、加速度、当前关节可提供的竖直加速度和工作空间。不满足时进入 hold，并在 `juggle.fault` 显示原因。
- 起球先产生较低的 seed apex，随后每次有效反弹逐步提高目标，直到 `target apex`。`apex ramp` 是每拍请求增量；若该拍在共享速度/加速度内不可达，控制器会同时下调本拍 apex 增量和横向回中比例，并在遥测中分别显示 requested/applied apex，而不是裁剪 MPC 后破坏碰撞终端条件。同一段球飞行中重新规划只允许保持或继续降低这个可行性比例，不会在临近碰撞时重新升级目标；确认一次新反弹后才重新开放下一拍的 requested profile。
- MPC 在预计撞击前锁定已验证的事件边界；到达事件附近后进入 `contact_wait`。默认允许 60 ms 等待相机看到“下降变上升”，期间不因球心毫米级越过预测接触面而立即 fault；超时仍未确认反弹才 hold。
- 后端只有在先实际收到一次 `OFF`，随后收到新的 `Start nonce` 时才接受启动。同一个 nonce 重复发送或跨模式残留都不会重新倒计时。Stop、切换模式、Reset、开启/关闭实机命令流、视觉/关节反馈过期、预测不可达或 MPC 失败都会停止；fault 后必须先关一次按钮再重新 Start。

### 网页参数

左侧 `Racket MPC — Shared` 是 tracking 与 juggling 真正共用的一套运行时参数：horizon、position/velocity/effort/smooth/posture 权重、笛卡尔速度/加速度上限和关节速度/加速度比例。`Compensation Params` 中的 `compMpc*` 仍然只是执行器延迟补偿，不能当作球拍轨迹 MPC。

左侧 `MPC Juggle Model / Start` 包含倒计时、最终/首跳 apex、逐拍增量、起球峰值加速度、恢复系数、切向传递、空气阻力、球半径、放球容差、最大球状态年龄和 `contact grace`。网页运行时会回显：状态、倒计时、放球拒绝原因、预测接触时间/位置、期望碰撞拍速、MPC cost/饱和、终端误差、反弹次数和 fault。

### 推荐实验顺序

1. 保持 `Send command` 关闭，选择 `MPC juggling`，确认命令姿态、`right_ee_site`、拍面法向和所有共享限幅。按钮保持 OFF。
2. 空拍开启 `Send command`，只验证移动到初始姿态、Stop、Reset、命令租约与急停；不要放球，也不要点击 Start。
3. 降低共享速度后重复阶段 1 相机跟踪，确认共享面板确实同时作用于两种模式。返回 `MPC juggling` 后再恢复经过批准的起球限制。
4. 把球静放在拍面中心，等待页面显示 `ready=yes`。首次把 countdown 保持为 3 秒，点击 Start 后立即准备急停；先只验证一次起球和离拍检测。
5. 起球稳定后，从较小 seed/target apex 开始验证第一次回落碰撞，再逐拍增加高度。根据实测入射/反弹速度标定 restitution 和 tangential transfer；不要用提高权重掩盖错误的碰撞参数。

可用环境变量预设物理参数，例如：

```bash
DEFAULT_CONTROL_MODE=mpc_juggle \
JUGGLE_COUNTDOWN_S=3.0 \
JUGGLE_SEED_APEX_HEIGHT_M=0.01 \
JUGGLE_TARGET_APEX_HEIGHT_M=0.10 \
JUGGLE_LAUNCH_PEAK_ACCELERATION_M_S2=16.0 \
JUGGLE_RESTITUTION=0.58 \
JUGGLE_CONTACT_GRACE_S=0.06 \
ENABLE_CAMERA_BALL=true \
ENABLE_ROS_BRIDGE=true \
ROS_PUBLISH_COMMANDS=true \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

`ROS_PUBLISH_COMMANDS=true` 仍只授予网页发送权限；`Send command` 和 `Start juggling` 都默认关闭，必须分别由操作者确认。

本实现的 first-exit 分段、碰撞 reset 与起球/周期控制分离主要参考：Kulchenko 与 Todorov 的 [First-exit MPC 球反弹实验](https://www.roboti.us/lab/papers/KulchenkoICRA11.pdf)、ETH Blind Juggler 的 [三维球拍反弹与负碰撞加速度稳定性](https://blindjuggler.ethz.ch/wp-content/uploads/2024/09/icra2009.pdf)、7-DoF 机械臂从静止接触开始的 [swing-up / periodic juggling 两阶段最优控制](https://arxiv.org/abs/2606.06704)，以及 Cassie 实机的 [拍面倾角横向调节](https://hybrid-robotics.berkeley.edu/publications/ECC2020_CassieJuggling.pdf)。论文参数不是本机标定值，网页默认值只用于保守起步。



horizon_s 0.04
step_s 0.005
margin_m 0.02
ori_margin 0.01


考虑到颠球过程中球的运动时间比较短，是不是用匀速卡尔曼滤波比重力卡尔曼滤波的效果会更好一点。

ANALYSIS=true        # 交互选择最近录制目录
ANALYSIS=latest ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh    # 自动选最新
ANALYSIS=/path/to/probe_recording_xxx ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
ANALYSIS_RUN_DIR=/path/to/probe_recording_xxx ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

让./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh支持深色模式，并通过网页端上方的一个按钮控制切换。


另外我想要系统性的寻找sim2real gap产生的原因，还需要补充哪些数据记录。注意在obs中有些观测维度是存在因果关系的，比如球相对于球拍的位置速度等，和关节位置速度有比较大的关系，或者可以说直接是由机器人右臂关节位置和速度决定的（因为ie实物和仿真使用同样的球观测来源）。因此对于sim2real gap产生的原因要追根溯源，因此可能需要补充记录其他信息。
这部分prompt忽略，调节rmp最大加速度参数

我能不能通过球预测模型将球状态更新频率从90Hz提升到200Hz。保留高球预测，但约束 handover。reacquisition 应使用创新门控、短时连续融合或 covariance-aware handover，目标是把每轴交接残差控制在训练随机化范围内，而不是在重新看到球的一帧硬切换。                                                                                                                                                   
整体来看，球观测还是影响策略表现的最大问题之一。现在球状态检测又加了很多补丁，也有很多新功能，但是可能稍微有些冗余了，你重新review一下球检测相关代码，从一个完整路径分析当前球检测是否合理。要满足球向上飞出视野、snesor深度失效、漏检、错检等情况下都能提供稳定准确的球状态检测，而且当从不同情况恢复到正常情况时也不能出现检测突变的情况。而且你要充分考虑检测帧率以及颠球是个高动态任务，一次颠球时间肯能就零点几秒，要充分考虑不同参数设置和窗口长度的设置是否合理。

而且还有个问题当起球颠球时physics velocity总是会给球一个z一个负的速度，但是刚开始起球时这是不合理的，因为球是从0突然开始向上运动，没有负向速度

我在实物上进行了分析，发现球拍姿态对于颠球影响非常大，而且非常容易导致SIM2REAL GAP，因此需要在仿真中就考虑设计专门的球拍姿态控制器，不过目前还没有确定合理的方案。我觉得只通过奖励函数约束可能不能实现很好的效果，是不是可以通过一个额外附加的、可部署的姿态稳定控制器来使得颠球时球拍姿态稳定。我觉得是不是执行器模型尽可能对齐比较好，因为实物上部署时可能会出现误差累积的情况，导致策略输出越来越不稳定，或者有什么比较好的解决方案吗？                                                                                                                                                                                                                           