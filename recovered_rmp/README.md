# 从 `mc_core` 恢复的 RMP 规划器

这个目录给出了可运行的 RMP 核心算法，而不只是反汇编记录。它恢复的是 Movax
控制器中名为 `Reactive Motion Planner` 的加速度级规划器：任务空间 PD/前馈产生期望
加速度，Jacobian 将任务拉回关节加速度空间，随后求解带关节安全边界的凸 QP。

对实物当前使用的 `RMP motion mode`，主入口是 `JointRMPPlanner`：上游只给 7 维位置
目标，规划器在 1 kHz 下生成 q/qd/qdd 三路命令。任务空间接口 `solve_task_rmp()` 也已
保留，可用于策略直接输出末端目标的实验。

## 已恢复的实际算法

原程序每个控制周期求解

```text
min  0.5 qddᵀ H qdd + gᵀ qdd
s.t. lower <= qdd <= upper
```

任务空间模式的期望加速度为：

```text
a*.linear  = kp e.linear
           + kd (linear_vel_ffd Vd.linear - V.linear)
           + linear_acc_ffd Vd_dot.linear

a*.angular = kp e.angular
           + kd (angular_vel_ffd Vd.angular - V.angular)
           + angular_acc_ffd Vd_dot.angular
```

其中平移、旋转误差分别先限制到
`max_velocity * dt * 200`，得到的两段加速度又分别按向量范数限制到配置的最大加速度。
低速时，程序还会把速度前馈系数乘以下列连续增益：

```text
translation: 0.25 + 15 speed,       speed < 0.05 m/s
rotation:    0.25 + 2.1428 speed,   speed < 0.35 rad/s
```

令 `r = a* - Jdot qd`，恢复出的 QP 矩阵是：

```text
H = JᵀJ + (vel_weight dt² + acc_weight) I
g = -Jᵀr
    + vel_weight dt qd
    + 4 pos_limit_weight |1 - psi_gain| (q - q_mid)
```

这等价于同时最小化任务加速度误差、下一周期关节速度和关节加速度，并以线性项将
冗余关节推向行程中点。若启用 7 自由度臂角，程序追加完整 Jacobian 的第七行：

```text
psi_dd* = clip(psi_kp wrap(psi_d-psi) - psi_kd psi_dot, ±max_psi_acc)
```

`psi_gain` 在算法奇异度低于 0.1 时为 0，在 `[0.1, 1)` 中等于奇异度，之后为 1。

关节目标模式则是：

```text
a* = clip(
       jnt_kp (q_target-q)
     + jnt_kd (joint_vel_ffd qd_target-qd)
     + joint_acc_ffd qdd_target,
     ±qdd_max)

H = (1 + acc_weight) I
g = -a*
```

无约束时因此有 `qdd = a*/(1+acc_weight)`，不是简单地直接返回 PD 输出。

实物关节位置目标在进入这个 QP 前，还依次经过以下已从机器码确认的运行时逻辑：

1. 新目标按 `qd_max * min(距上次目标的tick数, 100) * dt` 做逐关节位置变化限速；
2. Kalman 只在新目标到达的那个 tick 校正，保持目标期间只预测；超过 20 tick 没有
   新目标时，按 `clip(-qd/dt, ±qdd_max)` 主动减速；
3. 估计出的 q/qd/qdd 写入长度 10 的目标环形均值窗口；
4. QP 求解并积分后，原始 q/qd/qdd 再分别写入长度 35 的输出环形均值窗口。

QP 求出加速度后，`Plan` 使用下式更新内部原始轨迹：

```text
q_next  = q + (qd + 0.5 dt qdd) dt
qd_next = qd + dt qdd
```

复制程序中 `PlanTask` 的周期是字面量 `0.001 s`；200 Hz ROS/策略位置命令因而会在
5 个连续伺服周期中保持。最终送给低层控制器的 `axis_pos/axis_vel/axis_acc` 是三个
长度 35 窗口的均值，而不是未滤波的积分状态，也不是日志中上游 `policyCommand` 的
速度和加速度列。实机该路径允许原始加速度在正负上限间切换，然后靠输出窗口平滑；
YAML jerk 限制不参与此运动模式。

## 已接入 simcode5/MJX 的实物控制链

[`mjx_juggle_env.py`](../simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/mjx_juggle_env.py)
新增了 `recovered_rmp_motion_mode`。启用后的路径是：

```text
策略积分位置 / step_q_ref 记录位置
  -> 5 × 1 ms recovered joint RMP
  -> 每个子步一组 q/qd/qdd
  -> XML 低层 PD
  -> 5 × 1 ms MJX physics
```

RMP 与 MJX 是逐子步锁步的，不是先算完 5 次 RMP 再把最后一个位置保持 5 ms。该模式
会拒绝同时开启 actuator filter/FOPDT、逆补偿、额外 servo planner、plant delay 或实际
状态投影，确保“无执行器模型”不是一个只写在配置名里的假设。XML position actuator
仅作为低层 PD；RMP 速度通过 `(Kv/Kp)*qd_ref` 的等价位置项精确进入 PD，qdd 前馈可单独
标定，当前日志的最优默认值是 0。

直接构造环境：

```python
cfg = MjxJuggleConfig(
    recovered_rmp_motion_mode=True,
    recovered_rmp_jnt_kp_scale=1.0,       # 恢复 YAML: Kp=400
    recovered_rmp_jnt_kd_scale=1.0,       # 恢复 YAML: Kd=40
    recovered_rmp_controller_kp_scale=0.02,
    recovered_rmp_controller_kd_scale=0.10,
    actuator_cmd_filter=False,
    actuator_compensation_mode="none",
    enable_delay_conditioning=False,
)
```

快速 PPO 入口已经暴露相同模式：

```bash
cd simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim
python train_juggle_mjx_ppo.py \
  --recovered-rmp-motion-mode \
  --rmp-calibration-profile recovered_yaml_pd_fit
```

旧的 `record_new3_4_q_qd_qdd` profile 是在发现上述运行时逻辑之前拟合的，只保留作历史
对照，不再推荐给训练。现在优先使用恢复 YAML 原值（RMP Kp/Kd=`400/40`、目标窗口 10、
输出窗口 35、jerk 关闭）。在与 `record_new4/203833` 同次运行的 1 kHz DataTracer 上，
正式实现相对实机 `axis_pos/axis_vel/axis_acc` 的 3590 点 RMS 为：

```text
0.038381° / 0.010817 rad/s / 0.303445 rad/s²
```

三项均通过 `0.1° / 0.05 rad/s / 1 rad/s²` 门槛。这里验证的是 RMP 输出层；XML 理想 PD
之后的 MuJoCo 关节反馈是另一个辨识问题，不能把两种指标混为一谈。

截至 2026-08-13，同一 DataTracer 稳定段把最终 reference 与编码器 feedback 用每关节最优
下游延迟直接比较，仍为 `0.156158° / 0.078217 rad/s / 10.200634 rad/s²`。这不是 RMP
恢复误差，而是未记录的低层伺服/机械/反馈残差；无执行器模型的 RMP+XML-PD 路径尚未满足
物理反馈的三个硬阈值。完整结论、当前 MJX 留出回放和所需采集字段见
[`CALIBRATION_RESULTS.md`](CALIBRATION_RESULTS.md)。

## record_new3/4 回放

独立 MuJoCo 理想力矩 PD 回放：

```bash
python simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/replay_rmp_recording.py \
  <probe_recording目录或control_samples.jsonl> \
  --calibration-profile recovered_yaml_pd_fit \
  --output /tmp/rmp_replay
```

走完整 `MjxJuggleEnv` 训练路径的回放（需 `pingpong_mjx` 环境）：

```bash
python replay_rmp_recording_mjx.py <control_samples.jsonl> \
  --calibration-profile recovered_yaml_pd_fit \
  --output /tmp/rmp_mjx_replay
```

两者都会保存目标、实测、RMP q/qd/qdd、仿真 q/qd 和 JSON 指标。现有交叉记录结果与
准确性边界见 [`CALIBRATION_RESULTS.md`](CALIBRATION_RESULTS.md)。

批量回放后的拍面（`right_ee_site`）sim-vs-real 误差可用
`src/.../rl_sim/analyze_racket_sim2real_batch.py` 计算：实物侧对记录关节角做
MuJoCo FK + 原始 `ros.jointVelocity`，仿真侧对 MJX replay 关节角做同样算子，末端
线/角加速度用 Savitzky-Golay（窗口 15、阶 3）滤波求导，输出位置/姿态角/线速度/
角速度/线加速度/角加速度的逐轴 RMS 与范数 RMS。2026-08-20 全量 64 条记录的结果
见 `analysis_results/racket_sim2real_rmp_parallel_20260820.json` 与 `RMP.md`。

若有同次运行的 Movax DataTracer，先导出并直接验证恢复的 RMP 输出：

```bash
python3 -m recovered_rmp.extract_movax_datatracer <dat文件> \
  --output /tmp/right_arm_trace

python3 -m recovered_rmp.validate_rmp_datatracer \
  /tmp/right_arm_trace.npz <probe_recording目录> \
  movax/controller/configuration/model/user/spirit01_base_e_no_neck_hh.yaml \
  --include-feedback-reference-baseline
```

## JSON 配置接口（推荐接入方式）

`recovered_rmp/interface.py` 提供纯 NumPy 的便捷接口，用一个 JSON 文件统一维护
RMP 参数与 XML 低层 PD 参数，方便把 RMP 接入任意仿真环境。现成的配置是
[`configs/rmp_equiv_refine_baseline.json`](configs/rmp_equiv_refine_baseline.json)：
RMP 侧是恢复 YAML 原值（jnt Kp/Kd=400/40、窗口 10/35），PD 侧是 record_new3/4/5
replay 标定的 per-joint 缩放与输出延迟（18,19,19,18,18,18,18）。JSON 键缺省时回退
到 RightArm YAML 默认值；角度一律 SI（弧度）。

```python
from recovered_rmp import create_planner, drive_joint_rmp, load_rmp_config

config = load_rmp_config("recovered_rmp/configs/rmp_equiv_refine_baseline.json")

# 1) 直接拿 planner 自己控制 tick（config.rmp / config.limits 已就绪）
planner = create_planner(config)
planner.reset(q0, velocity=qd0, target_position=q_target0)
cmd = planner.step(q_target, target_updated=True)   # cmd.position/velocity/acceleration (7,)

# 2) 或整段目标轨迹一次跑完：200 Hz 目标，内部 1 kHz 推进
commands = drive_joint_rmp(targets, config, initial_position=q0, control_period=0.005)
```

`config.xml_pd` 保留低层 PD 的 kp/kv、per-joint 缩放、damping/armature 缩放和
qdd 前馈等字段，供 MJX 类环境直接读取；`config.output_delay_steps_per_joint` 是
RMP 输出到 PD 的纯传输延迟。完整示例见
[`examples/interface_usage.py`](examples/interface_usage.py)。注意：该接口只驱动
关节空间 RMP，不含 MuJoCo PD/物理；接入仿真时把命令送入你自己的低层 PD 即可
（对应 env 里的 `step_q_ref` 路径）。

## 在 MuJoCo 中调用

下面的接口把私有 `rkdl` 替换为 MuJoCo 自身的 FK/Jacobian。所有量都使用 SI：米、秒、
弧度。

```python
import numpy as np

from recovered_rmp import frame_error, load_movax_rmp_config, solve_task_rmp
from recovered_rmp.mujoco_adapter import site_kinematics

params, limits = load_movax_rmp_config(
    "movax/controller/configuration/model/user/spirit01_base_e_no_neck_hh.yaml",
    "RightArm",
)

# dof_indices 是 MuJoCo 中右臂 7 个 qvel/dof 下标。
kin = site_kinematics(model, data, "racket_site", dof_indices)
q = data.qpos[qpos_indices]
qd = data.qvel[dof_indices]
q_mid = 0.5 * (limits.q_min + limits.q_max)
error = frame_error(kin.position, kin.rotation, target_pos, target_rotation)

result = solve_task_rmp(
    q=q,
    qd=qd,
    q_mid=q_mid,
    limits=limits,
    jacobian=kin.jacobian,
    jacobian_dot_qd=kin.jacobian_dot_qd,
    error=error,
    current_twist=kin.twist,
    target_twist=np.zeros(6),
    target_acceleration=np.zeros(6),
    params=params,
    dt=control_dt,
)
qdd_command = result.qdd
```

完整的一步封装见
[`examples/right_arm_mujoco_step.py`](examples/right_arm_mujoco_step.py)。给现有
`MjxJuggleEnv` 使用时，它的归一化动作是：

```python
action = np.clip(
    qdd_command / (env.arm_acc_limit_rad_s2 * env.cfg.action_acc_scale),
    -1.0,
    1.0,
)
```

批量 MJX 训练可用 [`jax_backend.py`](jax_backend.py)。关节位置 RMP 的对角 QP 有解析解，
NumPy 与 JAX 的状态轨迹等价测试覆盖目标限速、事件驱动估计器和双层输出窗口。一般
任务空间 QP 为能被 `jax.jit`
编译，采用固定次数投影梯度，而不是原二进制链接的 ProxQP。典型用途有三种：

1. 用 RMP 从末端目标生成 `qdd` 专家轨迹，做行为克隆或预训练；
2. 策略输出末端目标/RMP 残差，由 RMP 负责运动学与关节限制；
3. 用 `joint_acceleration_bounds()` 把策略加速度投影进恢复出的安全包络。

## 配置与测试

`load_movax_rmp_config()` 会读取拷贝来的真实右臂参数，同时把 YAML 中的角度、角速度、
角加速度和角 jerk 从度制转为弧度制。当前右臂关键值是 `kp=324`、`kd=36`、
`jnt_kp=400`、`jnt_kd=40`、`vel_weight=30`、`acc_weight=0.001`。

运行验证：

```bash
cd /home/spirit-ai/pingpong_moz1
python3 -m unittest discover -s recovered_rmp/tests -v
```

测试覆盖 QP 公式、随机凸 box-QP 的独立枚举校验、误差/加速度限幅、关节安全边界、
Kalman 状态估计/超时减速、目标限速、NumPy/JAX 状态轨迹等价、全身任务组合和真实
YAML 单位转换。

## 恢复边界

以下内容是从符号、DWARF 类型和机器码逐项确认的：任务/关节参考加速度、`H/g`、
前馈调节、误差与加速度限幅、臂角支路、主要关节约束结构、参数布局以及 ProxQP
目标形式。

以下内容是为了脱离私有运行库而做的等价实现：

- `rkdl` FK、位姿差和 Jacobian 改由调用方/MuJoCo 提供；
- 原 ProxQP 换为独立的 box-QP active-set 求解器；
- 安全边界中优化后的分段二次求根写成同一制动不变量的闭式包络；
- Kalman 的 F/Q/R、初始 `P00=0.01` 及长度 10 的三路环形均值已恢复；Joseph 协方差
  更新是与标量观测公式代数等价、数值更稳定的写法；
- WBRMP 的防倾覆/平衡专用任务尚未冒充为已恢复逻辑。

完整证据、地址和可信度分级见 [`RECOVERY_EVIDENCE.md`](RECOVERY_EVIDENCE.md)。
