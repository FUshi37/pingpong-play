# MOZ1 底盘动力学辨识与仿真建模工具链 (`tools/base_id`)

本工具链依据 [docs/basetest.md](../../../../../../docs/basetest.md) 与真机 (`172.16.0.20`, `ROS_DOMAIN_ID=33`) 的实际 ROS 2 / `mc_core` 接口实现，用于在安全围栏下采集无球底盘激励数据，并离线拟合出可直接导入 MuJoCo/MJX 仿真的底盘动力学与拍面抖动参数。

---

## 一、真机 ROS 2 接口核查结论（来自 `172.16.0.20` 实测与 `mc_core` 源码）

1. **底盘控制存在两套互斥协议（均已在本工具链支持）**：
   - 当 `motion.yaml` 中 `use_new_robot_base_cmd: true`（**当前真机/仿真已启用的最终配置**）：
     - `MCControlNode` 订阅 **`/mx_base_cmd`**，消息类型为 **`mc_core_interface/msg/MechUnitValueArray`**（`BEST_EFFORT`），单条 `MechUnitValue(name="base", mode=1)`，`cartesian.frame_type` 需与控制器 `frame_type_` 一致（`0` 或 `1` 为速度控制，`2` 为世界系位置控制），由 `cartesian.twist.linear.{x,y}` 与 `cartesian.twist.angular.z` 下发速度。
     - 合同设置：`base_command.protocol: mech_unit_value_array`、`base_command.topic: /mx_base_cmd`、`mode: 1`。
   - 当 `motion.yaml` 中 `use_new_robot_base_cmd: false`（旧版接口）：
     - `MCControlNode` 订阅 **`/mx_base_vel_command`**，消息类型为 **`geometry_msgs/msg/Vector3`**（`BEST_EFFORT`），字段 `x = vx (m/s), y = vy (m/s), z = wz (rad/s)`。
     - 合同设置：`base_command.protocol: vector3_vel`、`base_command.topic: /mx_base_vel_command`。
2. **底盘固定 K 形刚体动捕（任意单点贴装、无万向锁、抗编号乱跳与短暂遮挡）**：
   - 针对实验室固定的 4 球 K 形刚体，默认模式为 `mocap.base_track_mode: rigid_body`。**无需贴在底盘中心、也无需水平平放**：直接贴在底盘本体外缘任意不被腰腿遮挡的平面或侧面（朝向动捕主相机群，因测试波形对称往返，单侧最大摆角仅 ±22°~±75°）。
   - **抗多 K 形刚体编号乱跳**：因场上存在多个相同 K 形刚体导致 UDP `opti_rb_<slot>` / `id` 逐帧互换（见 [docs/mocap_observation_range.md](../../../../../../docs/mocap_observation_range.md)），应通过 `mocap.base_rigid_body_observation_bounds_m` 用原始动捕 XYZ 范围框选底盘刚体（例如底盘高度 `Y` 明显低于台上固定刚体的 `0.82–1.74 m`，可设 `[null, [0.05, 0.65], null]`）。
   - **任意安装角度无万向锁**：自动用 `mocap_calibration_auto.txt` 做 Y-up→Z-up 相似变换，并以初始静止帧为基准做右乘消偏 $R_{\text{leveled}}(t) = R_{\text{robot}}(t) R_{\text{ref}}^T$，即使 K 形板 90° 竖贴在底盘侧壁也能准确提取 `yaw` 与底盘动态 `roll/pitch` 晃动（在串口 IMU 离线时自动合入拍面抖动统计）。
   - **自动偏心距标定 + 抗短暂遮挡**：离线拟合自动用 `step_wz` 闭式反解 K 形刚体到底盘旋转中心的偏心矢量 `lever_arm_xy_m` 并还原所有平动/圆弧轨迹到底盘中心；实时采集默认 `mocap.abort_on_dropout: false`，转向中偶发短暂遮挡不会误中断实验（200 Hz `/mx_robot_state` 始终做 100 ms 严格看门狗与围栏兜底）。
2. **高频反馈主通道（200 Hz）**：
   - **`/mx_robot_state`** (`mc_core_interface/msg/MechUnitValueArray`, `RELIABLE`, 实测 **200.05 Hz**)：同步包含 `name="base"` 的 `cartesian.twist`（由 4 轮全向正运动学计算）和 4 轮编码器 `joint.{position, velocity, effort}`，以及 `name="right_arm"` 的 7 轴 `joint.{position, velocity, effort}`。
   - **关键细节**：当控制器 `frame_type == 0` 时，`/mx_robot_state` 中 `base.cartesian.pose` 恒为 Identity `(0,0,0)`；因此采集节点默认使用 `base_pose_source: integrate_base_twist` 实时积分机体系速度作为软件地理围栏里程计（若开启外部动捕，则同时对动捕与积分里程计取最保守包络触发围栏）。
3. **机械臂并发通道（用于 `s5_arm_concurrency`）**：
   - 当前真机 `motion.yaml` 设置了 `use_new_robot_cmd: true`，因此 `MCControlNode` 订阅 **`/mx_robot_cmd`** (`MechUnitValueArray`, `name="right_arm"`, `mode=0` 关节位置)，且明确拒绝在 `/mx_robot_cmd` 中发 `base`。本工具同时兼容旧版 `/mx_mix_command` (`MechUnitCmdArray`)。
4. **真机开跑前务必注意的 4 项现场配置发现**：
   - **测试前置姿态与腰腿阻抗模式（务必先切到 `VRMOV` 并开启 `LegWaist` 阻抗）**：若腿腰处于全零直立位姿，高加速横向 `vy` 阶跃会触发 `mc_core` 倾覆急停（`SAFETY_STATE_OVERTURN_STOP`, `safety_status=4`）；开跑前应先上电移动至 `VRMOV`（`LegWaist=[0, 60, -90, 30, 0, 0]°`）并打开 `LegWaist`（`mu_idx=1`）阻抗控制模式（`control_mode[1]=3` 或 `4`，可通过合同 `safety.require_legwaist_impedance_mode: true` 强制启动前检查）。
   - **底盘速度/加速度上限与四轮速度多面体约束**：在 RMP 跟踪模式（`/mx_base_cmd`）下，`base_vel_override=0.1` **不缩放**底盘速度（仿真验证实测增益 `1.000`）；单轴线性速度上限由 `Base.rmp_params.linear_limits` 限幅在 **`0.70 m/s`**，单轴加速度上限由 `max_acc=1000 deg/s²` 决定（约 **`2.45–2.48 m/s²`**，角加速度约 **`5.06 rad/s²`**）。同时，`MonitorTask` 对单个全向轮施加 **`5.445 rad/s`** 速度上限（故障阈值 `5.990 rad/s`），因此斜向/边走边转组合机动必须满足四轮多面体约束 $\frac{|v_x|+|v_y|}{\sqrt{2}\cdot 0.10} + 3.427\,|\omega_z| \le 5.445\text{ rad/s}$（`waveforms.py` 已内置 `<= 4.95 rad/s` 安全包络）。
   - **遥操发布者冲突**：确认 `MozTeleopRobotNode` 未在 `/mx_base_cmd` 上并发发包。
   - **IMU 与时钟偏置**：`/imu_states` 仅在底盘串口 IMU 打开时发布（当前 0 Hz，合同默认 `required: false`）；同时真机 `header.stamp` 与上位机墙钟存在常数偏置（实测约 `-1.343 s`），离线拟合脚本会自动检测、剥离并写入 `DATA_AUDIT.json`。

---

## 二、标准执行流程

### 1. 只读接口审计（零运动）
```bash
source /opt/ros/humble/setup.bash
source /home/spirit-ai/ros_interface-v2.1.2.3/install/setup.bash
python3 src/pingpong_moz1/pingpong_controller/tools/base_id/inspect_base_interface.py
```

### 2. 填完 `UNRESOLVED_REQUIRE_OPERATOR_FILL` 后先跑 `--dry-run`
```bash
python3 src/pingpong_moz1/pingpong_controller/tools/base_id/run_base_id_experiment.py \
  --stage s1_kinematic_low --dry-run
```
如需切换至新版 `/mx_base_cmd` (`mode=1`, `twist`) 协议，只需修改合同 YAML 或通过 `--set` 覆写：
```bash
python3 src/pingpong_moz1/pingpong_controller/tools/base_id/run_base_id_experiment.py \
  --stage s1_kinematic_low --dry-run \
  --set base_command.protocol=mech_unit_value_array \
  --set base_command.topic=/mx_base_cmd \
  --set base_command.msg_type=mc_core_interface/msg/MechUnitValueArray
```

### 3. 现场分阶段 `--live` 采集（全程手握硬件 E-Stop）
```bash
CONFIRM_MOZ1_BASE_ID_LIVE=YES \
python3 src/pingpong_moz1/pingpong_controller/tools/base_id/run_base_id_experiment.py \
  --stage s0_idle_noise --live
```
随后按 `s1_kinematic_low` → `s2_step_ramp_med` → `s3_step_brake_high` → `s4_combined_turn` → `s5_arm_concurrency` 逐级推进。

### 4. 离线动力学辨识与出图（拆场前必做）
```bash
python3 src/pingpong_moz1/pingpong_controller/tools/base_id/fit_moz1_base_dynamics.py \
  outputs/base_id/base_id_* --out outputs/base_id/summary_$(date +%Y%m%d_%H%M%S)
```
输出产物：`moz1_base_identified_params_v1.json`、`DATA_AUDIT.json`、`BASE_ID_REPORT_CN.md`、`fig_moz1_base_id_summary.pdf/.png`。
