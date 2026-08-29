# RMP 使用说明（RMP Usage Guide）

本文档面向**把 recovered RMP 接入自有仿真环境**的使用者。算法背景、恢复证据和标定状态见
[recovered_rmp/README.md](recovered_rmp/README.md)、[RECOVERY_EVIDENCE.md](recovered_rmp/RECOVERY_EVIDENCE.md)、
[CALIBRATION_RESULTS.md](recovered_rmp/CALIBRATION_RESULTS.md)；批量回放与末端误差结论见 [RMP.md](RMP.md)。

## 1. 是什么

`recovered_rmp/` 是 Movax 机器人 `Reactive Motion Planner` 运动模式的纯 Python 复现：
上游只给关节位置目标（200 Hz），RMP 内部在 1 kHz 伺服节拍产生 q/qd/qdd 三路命令，
再送低层位置 PD。整套链路（含目标限速、Kalman 目标估计、双层环形均值、凸 QP、输出
纯传输延迟）已按机器码逐一确认，与实机 `axis_pos/axis_vel/axis_acc` 对齐（见 README
验证段）。

## 2. 依赖

| 用途 | 依赖 | 环境 |
| --- | --- | --- |
| 接口层 + planner（纯 NumPy） | numpy（>=1.23） | 任意 Python |
| 批量 MJX 回放/训练 | jax + mujoco-mjx | `conda activate pingpong_mjx` |
| Movax YAML 读取 | pyyaml | 同上 |
| 末端 sim2real 误差分析 | mujoco + scipy | 同上 |

`recovered_rmp` 不依赖 ROS、不依赖私有 `rkdl`：任务空间接口的 FK/Jacobian 由调用方
或 MuJoCo 提供。

## 3. 快速开始（纯 NumPy，不加载物理）

```python
from recovered_rmp import create_planner, drive_joint_rmp, load_rmp_config

config = load_rmp_config("recovered_rmp/configs/rmp_equiv_refine_baseline.json")

# 方式 A：自己控制每个伺服 tick（1 kHz）
planner = create_planner(config)
planner.reset(q0, velocity=qd0, target_position=q_target0)
cmd = planner.step(q_target, target_updated=True)  # JointRMPCommand
cmd.position / cmd.velocity / cmd.acceleration     # 各 (7,)

# 方式 B：整段 200 Hz 目标一次跑完（内部自动按 1 kHz 推进，每个目标保持 5 tick）
commands = drive_joint_rmp(targets, config, initial_position=q0, control_period=0.005)
```

`target_updated=True` 只在新目标到达后的**第一个**伺服 tick 传；其余保持 tick 传
`False`（与实机 `Plan` 一致）。`drive_joint_rmp` 已自动处理。

## 4. 接入自己的仿真环境

RMP 只负责关节空间参考命令；物理积分和低层 PD 是仿真侧的事。推荐结构：

```
q_target (你的策略/记录)
  -> JointRMPPlanner.step()          # 接口层，JSON 参数
  -> 命令经 output_delay_steps_per_joint 延迟（若你的 PD 链未建模该延迟）
  -> 你自己的低层 PD：tau = kp*(q_ref - q) - kv*qd
     （kv 用 Kv/Kp*qd_ref 的等价位置项把 RMP 速度送进 PD，见 README「已接入 simcode5/MJX」段）
  -> 物理积分
```

如果你用的是 MuJoCo/MJX 类 `position` actuator，可直接取 `config.xml_pd` 里的
kp/kv/缩放/前馈来建 PD；`xml_pd.kp_scale_per_joint` 等字段与 `MjxJuggleEnv` 的
`recovered_rmp_controller_kp_scale_per_joint` 语义一一对应。接任务空间接口时：

```python
from recovered_rmp import frame_error, load_rmp_config, solve_task_rmp
from recovered_rmp.mujoco_adapter import site_kinematics
# kin = site_kinematics(model, data, "right_ee_site", dof_indices)
# error = frame_error(kin.position, kin.rotation, target_pos, target_rot)
# result = solve_task_rmp(q, qd, q_mid, config.limits, kin.jacobian,
#                         kin.jacobian_dot_qd, error, kin.twist, 0, 0,
#                         config.rmp, config.dt)
# 归一化动作：np.clip(result.qdd / (acc_limit*action_acc_scale), -1, 1)
```

## 5. JSON 配置

单文件维护所有可调参数：`recovered_rmp/configs/rmp_equiv_refine_baseline.json`，
结构由 `recovered_rmp/configs/rmp_config_schema.json` 约束。四块：

- `planning`：伺服周期 dt（默认 1 kHz）、安全边界/加加速度开关、目标更新延迟 3 tick、
  输出纯传输延迟 `output_delay_steps_per_joint`（当前 18,19,19,18,18,18,18）；
- `rmp`：RMP 增益（恢复 YAML：jnt Kp/Kd=400/40、kp=324、kd=36）、目标/输出窗口
  10/35、估计器方差、前馈系数，以及可选 per-joint 缩放字段；
- `joint_limits`：七关节 q/qd/qdd/qddd 限位（弧度制）；
- `xml_pd`：低层 PD 的 kp/kv 原始值、per-joint 缩放、dof damping/armature 缩放、
  qdd 前馈（标定于 record_new3/4/5）。

规则：键缺省时回退 RightArm YAML 默认；角度一律 SI（弧度）；per-joint 缩放列表
为空 = 不缩放；`load_rmp_config(..., apply_per_joint_scales=True)` 才会把
`rmp.jnt_*_scale_per_joint` 折进 planner 增益（默认不折，匹配实机标量增益语义）。
改参数只改 JSON，replay 命令行与接口层同步生效。

## 6. 批量回放与验证

```bash
cd simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim
conda activate pingpong_mjx
JAX_PLATFORMS=cpu python batch_rmp_replay_mjx.py \
  --model <workspace>/src/pingpong_moz1/pingpong_controller/tools/rl_sim/moz1_rmp_equiv.xml \
  --record-root <workspace>/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings \
  --include-group record_new3 --include-group record_new4 --include-group record_new5 \
  --output-root <output-dir> --batch-size 8 \
  --calibration-profile recovered_yaml_pd_fit \
  <RMP.md 中 refine-baseline 的 per-joint 参数>
```

注意事项：CPU 后端稳定（GPU 段错误）；`discover_recordings` 的 glob 偶发段错误，
重试即可（脚本自动复用已完成报告）；`record_new4/205641` 因数据缺失被跳过属正常。

末端 sim2real 误差分析（位置/姿态/速度/加速度/角速度/角加速度）：

```bash
python3 src/pingpong_moz1/pingpong_controller/tools/rl_sim/analyze_racket_sim2real_batch.py \
  <batch-output-dir> --output analysis_results/racket_sim2real_rmp.json
```

口径：实物 = 记录关节角 FK + 原始 `ros.jointVelocity`；仿真 = MJX replay 关节轨迹
同样算子；加速度 = 末端速度 Savitzky-Golay（窗 15、阶 3，缺口不跨段）。2026-08-20
结论见 RMP.md（新3/4 组位置范数 RMS 约 5–6 mm，new5 约 10 mm，剔除两条发散记录后）。

## 7. 测试

```bash
cd /home/spirit-ai/pingpong_moz1
python3 -m unittest discover -s recovered_rmp/tests -v   # 25 项：QP、Kalman、等价性、接口层
```

## 8. 已知边界

- 接口层只驱动关节空间 RMP；任务空间接口需调用方提供 FK/Jacobian。
- 无执行器模型的 RMP+XML-PD 路径**尚未**满足物理反馈三阈值（q/qd/qdd 与编码器
  feedback 的差距仍是未记录的低层伺服/机械残差），末端误差当前为毫米~厘米级，
  未达全场毫米级，勿宣称无 sim2real gap。
- 每条 replay 报告的 `racket_metrics` 是仿真轨迹本身，不是误差；误差由
  `analyze_racket_sim2real_batch.py` 从关节轨迹另算。
- WBRMP 防倾覆/平衡任务尚未恢复。
