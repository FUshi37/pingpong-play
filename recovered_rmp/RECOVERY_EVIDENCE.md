# RMP 恢复证据与可信度

## 二进制来源

分析对象均为用户拷贝到本机的文件，分析过程只做静态读取，没有启动机器人控制程序。

| 文件 | SHA-256 | 状态 |
|---|---|---|
| `movax/controller/mc_core` | `7e0097ab618bf6789eb977626a7259b7d901fe63b4087b965e9d54b0ad0c1e2c` | 当前启动脚本使用；ELF section-header 区域损坏，但 loadable text/rodata 可读 |
| `movax/controller/mc_core_bak` | `f6179ae402e80d7e1a287f452fb5a3dd7f34c2ce3c93526c66778dc4d1f7f8a2` | Build ID `334de9f35edb7fabcfdde30a94b1ef66dd4f961e`；完整 DWARF；未 strip |

当前文件与 `mc_core_bak` 的 RMP `.text` 函数高度一致。多个函数的机器码首段逐字节相同，
地址统一相差 `-0x2c0`，因此用备份的 DWARF 注释当前 loadable code 是可靠的版本映射，
而不是把完全不同版本的源码信息硬套上去。

| 函数 | `mc_core_bak` | 当前 `mc_core` | DWARF 源位置 |
|---|---:|---:|---|
| `IndepRMPPlanner::Plan` | `0x7cac50` | `0x7ca990` | `independent_rmp_planner.cpp:179` |
| `WBRMPPlanner::Plan` | `0x7de5b0` | `0x7de2f0` | `wbrmp_planner.cpp:538` |
| `WheelBaseReactiveMotionPlanner::Plan` | `0x821630` | `0x821370` | `wheel_base_reactive_motion_planner.cpp:231` |
| `ManipulatorReactiveMotionPlanner::Plan` | `0x9fc1d0` | `0x9fbf10` | `manipulator_reactive_motion_planner.cpp:288` |

## 直接确认的函数和数据结构

机械臂恢复主要来自 `mc_core_bak`：

| 函数 | 地址 | 源行入口 | 恢复内容 |
|---|---:|---:|---|
| `TargetCmdFilter` | `0x9f1e00` | — | 目标位姿/速度/加速度滤波流程 |
| `UpdateJointTarget` | `0x9f79f0` | — | 新目标速度包络、关节位置裁剪和更新标志 |
| `UpdateJointQPParams` | `0x9f7520` | 845 | 关节 PD/前馈、限幅、`H/g` |
| `UpdateIneqConstrains` | `0x9f86d0` | 779 | 位置/速度/加速度预测边界与失败判据 |
| `UpdateTaskQPParams` | `0x9f93b0` | 876 | 位姿误差、Cartesian PD/前馈、Jacobian 拉回、`H/g` |
| `Plan` | `0x9fc1d0` | 288 | 状态更新、QP 更新/求解、限制检查、命令输出 |

`PlanTask` 在当前二进制 `0x5db165` 从 rodata `0x19d02c0` 载入 double 字面量
`0.001`，对应 `plan_task.cpp:41` 的规划周期。`Plan` 求解后明确执行
`q_next=q+(qd+0.5*dt*qdd)*dt` 与 `qd_next=qd+dt*qdd`，随后写入三个长度
`filter_length=35` 的输出环；环均值才进入
`SysAxisControlCommand.axis_pos/axis_vel/axis_acc`。因此 q/qd/qdd 的运行时层不是根据
常见轨迹生成器猜出的包装，而是实际硬件命令语义的一部分。

本仓库实物节点的 `_make_unit_cmd` 只填充 `MechUnitCmd` 的关节位置并设置
`use_jnt=True`；记录中的 `policyCommand` qd/qdd 属于这个位置生成器的上游积分状态，
不能拿来冒充 `mc_core` 最终发送的 qd/qdd。

DWARF 中 `ManipulatorReactiveMotionPlanner` 明确包含 `proxsuite::proxqp::dense::QP<double>`、
`H/A/C/g/b/lb/ub`、任务与关节目标三阶轨迹、`J/Jdot`、全部关节限制、状态估计器和滤波器。
ProxQP 的标准形式与机器码写入完全一致：

```text
min 0.5 xᵀHx + gᵀx
A x = b
lb <= C x <= ub
```

独立机械臂路径的有效不等式是 `qdd_lower <= qdd <= qdd_upper`。全身路径保留全局
`A/C`，用于多机械单元的统一 QP。

`spirit_ai::base::RMPParams` 的二进制尺寸是 288 字节，关键偏移为：

| 偏移 | 字段 | 偏移 | 字段 |
|---:|---|---:|---|
| 0 | `kp` | 8 | `kd` |
| 16 | `jnt_kp` | 24 | `jnt_kd` |
| 32 | `psi_kp` | 40 | `psi_kd` |
| 48 | `angle_gap` | 72 | `vel_weight` |
| 80 | `acc_weight` | 88 | `pos_limit_weight` |
| 96..128 | 线/角速度、加速度和 `max_psi_acc` | 136..176 | 关节、线、角速度/加速度前馈 |
| 184,188 | 两个滤波长度 | 192 | `arm_weight_gain` |
| 200..240 | 三组过程/测量方差 | 248..250 | 三个布尔开关 |
| 256 | `check_limit_eps` | 264 | `damp_gain` |
| 272,273 | 停止/连续检查开关 | 280 | `continuous_check_coeff` |

这些字段与真实 YAML 的 `rmp_params` 一一对应，不是根据名字猜出的新参数。

## 公式恢复的机器码证据

### 关节模式：A级（直接确认）

`UpdateJointQPParams` 先计算并逐关节截断：

```text
a* = jnt_kp(q_track-q)
   + jnt_kd(joint_vel_ffd qd_track-qd)
   + joint_acc_ffd qdd_track
```

随后将 `H` 置为单位阵、对角线加 `acc_weight`、将 `g` 写为 `-a*`，并复制预测的
`qdd_lower/qdd_upper`。[`core.py`](core.py) 的 `compute_joint_reference()` 和
`build_joint_qp()` 是这段机器码的逐式翻译。

### 任务模式：A级（直接确认）

`UpdateTaskQPParams` 的局部变量由 DWARF 给出为 `X/V/Psi/Psi_dot/J/Jdot/Jdqd/Ve/Vr_dot`。
机器码进一步确认：

- 平移与旋转误差使用 `max_vel * period * 200.0` 范数截断；
- 低速前馈常数为 `0.05, 15, 0.25` 和 `0.35, 2.1428, 0.25`；
- 线/角参考加速度使用独立的 PD 和加速度前馈，并分别作范数截断；
- `Jdot qd` 从参考加速度中扣除；
- `H=JᵀJ`，随后加 `vel_weight*dt² + acc_weight`；
- `g=-Jᵀ(a*-Jdot qd) + vel_weight*dt*qd`；
- 行程中点项是机器码中的字面常数 `4.0 * pos_limit_weight * abs(1-psi_gain)`。

[`core.py`](core.py) 的 `compute_task_reference()` 与 `build_task_qp()` 是直接翻译。

### 关节约束：B级（结构直接确认，根式等价重写）

`UpdateIneqConstrains` 确认了以下顺序：

1. 初始化为物理 `±qdd_max`；
2. 使用 `q_min/q_max/angle_gap/q/qd/qd_max/qdd_max/dt` 生成预测安全速度区间；
3. 内部采用分段二次根，含机器码常数 `0.998, -4, -0.5, -2, 1e-9`；
4. 与 `±qd_max` 和一步可达速度 `qd±qdd_max*dt` 求交；
5. 用 `(v_bound-qd)/dt` 转为加速度边界；
6. 若上下界交叉小于 `1e-4`，以中点 `±1e-8` 修复，否则记录错误并返回失败。

当前 Python 用 `sqrt(2*0.998*a*distance)` 表达同一个停止距离不变量，避免照抄编译器展开的
区间求交代码。失败与微小交叉修复行为保留。因此它适合仿真，但不能宣称每个浮点舍入位
都与原 C++ 一致。

### 目标接收、状态估计和双层滤波：A级公式，B级浮点/容器细节

`UpdateJointTarget(0x9f79f0)` 对新位置目标先执行：

```text
allowance = qd_max * min(ticks_since_last_target, 100) * dt
accepted += clip(raw_target - accepted, -allowance, allowance)
accepted  = clip(accepted, q_min, q_max)
```

`Plan(0x9fc1d0)` 不会在每个 1 ms tick 用保持值反复校正 Kalman：新目标标志存在时只
调用一次 `JntStatesEstimator::Update`；无新目标的前 20 tick 调用 `Predict`；之后调用
`Deceleration(0x9f06c0)`，把加速度状态置为
`clip(-velocity/dt, ±qdd_max)`（速度绝对值不超过 `1e-6` 时置零），再做正常预测。

`SingleJointKalman` 的 DWARF 字段是 `x: Vector3d`、`P/F/Q: Matrix3d`、`H:
RowVector3d`、`R: double`、`K: Vector3d`；初始化局部变量包含 `dt` 到 `dt^5`，确认是
位置/速度/加速度常加速度模型。初始化机器码进一步确认：

```text
F = [[1, dt, 0.5 dt²], [0, 1, dt], [0, 0, 1]]
Q = process_var *
    [[dt⁵/20, dt⁴/8, dt³/6],
     [dt⁴/8,  dt³/3, dt²/2],
     [dt³/6,  dt²/2, dt]]
P = diag(0.01, 0, 0)
R = measurement_var
```

`TargetCmdFilter` 把 Kalman 得到的 q/qd/qdd 分别写入三个环形矩阵，qd 在写入前按关节
最大速度裁剪，然后按 `target_filter_length=10` 求均值。QP 后的原始积分 q/qd/qdd 又
分别写入三组 `filter_length=35` 环，完整窗口的均值才是下发命令。Reset 会填满两层
窗口。实际 DataTracer 中输出加速度相邻变化达到 `2*qdd_max/35`，同时证明 raw qdd 可从
负上限跳到正上限、该运动模式没有再施加 YAML jerk 边界。Python/JAX 使用 Joseph 形式
更新协方差；对标量位置观测它与原更新代数等价，但最后几个浮点舍入位不宣称逐位相同。

## 全身 WBRMP 的边界

WBRMP 的全局变量、每机械单元索引、任务/关节/轮式底盘更新函数、统一 ProxQP 和约束更新
已经确认。它与独立机械臂使用相同任务参考和 Jacobian-QP 形式。当前
[`whole_body.py`](whole_body.py) 提供把已经映射到同一全局 `qdd` 的任务度量与向量相加的
仿真实现。

尚未完成逐分支恢复的是 `BalanceTask` 中的质心/DCM/防倾覆策略，以及机器人专有的机构索引
和闭链约束。它们不影响固定底座右臂乒乓训练；若要训练底盘、腿腰和双臂共同运动，需要继续
恢复这些支路，不能把当前通用任务叠加器当成完整原厂防倾覆控制器。

## “实际算法”的准确含义

原 C++ 文本没有嵌在 ELF 中，无法恢复注释、变量原名以外的排版和模板源码；但核心控制语义
可以从未 strip 的函数、DWARF 类型/行表和机器码恢复。这里交付的是相同的控制方程、参数、
状态和约束含义，以及独立验证过的可运行实现。数值库、私有运动学库和少量优化后不可辨认的
边界记账以兼容实现替代，并已在本文逐项标出。
