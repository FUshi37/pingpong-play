# V91 Sim2Real 近 1:1 数字镜像 V3 分析

## 结论

V3 推荐使用 **command-driven sim mirror**：实物策略产生的同一组 `q_ref` 一路继续只发送给真实机器人，另一路只读复制给 sim mirror。mirror 使用辨识后的 recovered-RMP/执行器参数推进 MuJoCo，不向实物控制链回写任何量。这样既保留独立执行器和接触物理，又消除两个 actor 因微小状态差异而产生的闭环分岔。

三轨迹图片中的绿色曲线是该 command-driven V3 mirror；橙色仍是记录中的原始独立 sim mirror，蓝色是实物。所有曲线均为无 marker 的细实线。

## 三轨迹误差

| 时间窗 | 信号 | 原始 sim mirror | V3 command-driven mirror | 改善 |
|---|---|---:|---:|---:|
| full | 关节角 | 0.047440 rad | 0.003467 rad | 92.7% |
| full | 关节角速度 | 1.132231 rad/s | 0.064357 rad/s | 94.3% |
| full | 球拍位置 | 0.026862 m | 0.001983 m | 92.6% |
| full | 球拍速度 | 0.679084 m/s | 0.037275 m/s | 94.5% |
| post_warmup | 关节角 | 0.048014 rad | 0.002324 rad | 95.2% |
| post_warmup | 关节角速度 | 1.155064 rad/s | 0.050874 rad/s | 95.6% |
| post_warmup | 球拍位置 | 0.027358 m | 0.001507 m | 94.5% |
| post_warmup | 球拍速度 | 0.692860 m/s | 0.031077 m/s | 95.5% |
| held_out_14_s_to_end | 关节角 | 0.044523 rad | 0.002273 rad | 94.9% |
| held_out_14_s_to_end | 关节角速度 | 0.981155 rad/s | 0.053138 rad/s | 94.6% |
| held_out_14_s_to_end | 球拍位置 | 0.023890 m | 0.001406 m | 94.1% |
| held_out_14_s_to_end | 球拍速度 | 0.586191 m/s | 0.032673 m/s | 94.4% |

## V2 → V3 固定同一 q_ref 识别结果

| 时间窗 | 指标 | V2 | V3 | 改善 |
|---|---|---:|---:|---:|
| post_warmup | 关节角 | 0.002348 rad | 0.002324 rad | 1.0% |
| post_warmup | 关节角速度 | 0.051286 rad/s | 0.050874 rad/s | 0.8% |
| held_out_14_s_to_end | 关节角 | 0.002299 rad | 0.002273 rad | 1.2% |
| held_out_14_s_to_end | 关节角速度 | 0.053594 rad/s | 0.053138 rad/s | 0.9% |

V3 相对 V2 的数值收益较小但在 14 s+ 留出段仍同向，说明 V2 已接近当前 RMP 结构的执行器辨识极限。三轨迹中的大幅改善主要来自 command-driven 架构消除了独立 actor 状态分岔，而不是把该收益错误归功于几个新增参数。

## V3 mirror-only 执行器参数

- output delay J1–J7：`[13.875, 13.0, 11.5, 14.25, 12.75, 12.625, 13.0] ms`；允许 1 ms 相邻 RMP 输出间线性插值。
- MuJoCo position actuator Kp scale：`[0.85, 0.85, 0.85, 1.15, 1.15, 1.15, 1.15]`。
- MuJoCo position actuator Kv scale：`[1.15, 1.15, 1.15, 1.0, 0.85, 0.85, 0.85]`。
- RMP joint acceleration feedforward：`0.05`。
- J4 acceleration limit scale：`0.8`。
- low-level qdd feedforward：`[1.25e-05, -5e-05, 1.25e-05, 2.5e-05, -3.7500000000000003e-05, -3.7500000000000003e-05, -2.5e-05]`。
- dof damping relative scale：`[1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5]`。
- dof armature relative scale：`[0.5, 0.5, 0.75, 1.25, 1.5, 1.5, 1.5]`。

这些值只能作为 sim mirror override，不能写回 V91 checkpoint 或实物共享 `env_cfg`。实物接口仍只发布七维 `jnt_pos=q_ref`。

## 球拍接触层方案

Drag-ball 记录没有自由球—球拍冲击，不能用来再次拟合接触参数。接触层直接采用独立的 record_new3/4 碰撞辨识结果，并保留 record_new5 为留出验证：

- ball mass：`0.004 kg`；radius：`0.02 m`。
- pair friction：`0.35`。
- solref：`0.003 0.152540`。
- solimp：`[0.9, 0.995, 0.001]`；margin：`0.001 m`；condim：`3`。
- 训练随机化建议：mass `[0.0036, 0.0044] kg`，solref damping `[0.1, 0.24]`。

由于记录没有球自旋观测，胶皮摩擦与入射自旋仍不可分辨；MJX 与 CPU MuJoCo 的高速偏心接触也不是逐步完全等价。因此接触层不能宣称数学上的完全 1:1。

## 推荐运行结构

```text
camera / joint_states
          │
          ▼
      real V91 actor
          │ q_ref
          ├────────► 实物控制器（只收关节角）
          │
          └────────► command-driven sim mirror
                         ├─ V3 recovered-RMP/actuator
                         └─ 已辨识 ball-racket contact
```

自由运行的独立 actor mirror 仍可作为策略稳定性诊断模式，并继续使用 V2 闭环参数；不能把 V3 的执行器最优参数直接用于该模式。本次复核中 V3 参数的自由闭环表现变差，验证了两个模式必须分开。

## 泛化与上线边界

- 14 s+ 是同一次记录内的严格时间留出段，不等价于跨日期、跨温度和跨负载验证。
- V3 相对 V2 只改善约 1%，部分 Kp/Kv、damping、armature 比例落在本轮搜索边界；它们应视为离线候选，而不是已经唯一辨识的实物常数。
- 在至少两条新的 drag-ball 记录上复核前，不应自动替换在线 mirror 的默认参数。即使启用，也必须保持 `sim_mirror_only` 和禁止回写实物控制。
- 图中的约 95% 总改善包含 command-driven 架构收益；参数本身的独立增益以 V2→V3 表为准。
- 本轮已实现并测试 mirror 内部的分数毫秒延迟/Kp/Kv 参数能力，但在线 probe 尚未自动加载候选 JSON，也尚未把实物 `q_ref` 分叉接入 mirror；当前结果是确定性离线 replay 验证，不会改变现有部署。

## 图片与产物

- `01–03`：关节角、角速度、角加速度。
- `04`：共享 q_ref/action 与原始独立 mirror actor 输出。
- `05–10`：球拍位置、速度、加速度、Euler 角、角速度、角加速度。
- `11`：原始 mirror 与 V3 command-driven mirror 的误差时间序列。
- `proposed_sim_mirror_rmp_override_v3.json`：仅供 mirror 使用的执行器参数。
- `sim_mirror_digital_twin_v3.json`：执行器、接触和架构的组合方案。
