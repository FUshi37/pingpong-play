# GPU1 V153 QACC 实物部署与训练差异说明

本文对应用户从已安全停止的 GPU1 V153 训练中选出的 best checkpoint。它用于
分阶段实物部署实验，不表示 V153 已完成全部 66 个 curriculum stage，也不表示
已经通过实物验证。

## 1. 模型身份

| 项目 | 值 |
| --- | --- |
| profile | `goal_d455_sport_taskspace_record_new3_sim2real_fixed_base_ball4g_dual_domain_homotopy_v153` |
| stage | `record_new3_sim2real_v153_b2_b3_energy_p025_60hz`（index 59） |
| stage/global update | `441 / 1092` |
| true step | `14841511936` |
| seed | `82731` |
| actor / critic / action | `67 / 279 / 7` |
| checkpoint | `gpu1_v153_best_step14841511936.pkl` |
| SHA-256 | `a02b0421029e808b84ea0ee5cf615b3e65c6df3bc28c0f4766ad9953c0e85176` |

仓库路径：

```text
pingpong_controller/outputs/rl_sim/
selected_best_models_and_normal_reset_videos_20260901/gpu1_v153/
gpu1_v153_best_step14841511936.pkl
```

`gpu1_v153_qacc_reference.py` 先认证 SHA-256，再校验 checkpoint identity、
67/279/7 维度、单 actor 网络、QACC/command-state 合同、球时间戳、训练 actuator
参数、固定底座和球质量合同，并检查零 observation 的 actor golden-vector。

## 2. 能否直接合并到 simcode5

结论：可以以 simcode5 为基础新增独立 V153 profile，但不能在未修改配置/逻辑时
直接替换 simcode5 权重。

相对已提交 GPU1-QACC/V5 模型提交
`4f47e6d3c67917acd6a9b05162a488c91b319597` 和训练 companion
`69b9bf3442f5e9473bb291e0fa9439d0f15a53d5`，V153 的主要实物接口仍相同：

- 67-D actor、7-D QACC、deterministic mean；
- 200 Hz，`qacc -> dq_cmd -> q_cmd` command-state 积分；
- 每 tick 发布当前 `q_cmd`；45 ms command history 只用于 observation；
- 不在实物侧运行拟合执行器、XML PD、补偿器或额外 planner/RMP；
- reset/reconnect 时原子设置 `q_cmd=q_measured`、`dq_cmd=0`、previous action=0，
  并用当前 q 填满 observation-only buffer。

但 V153 继承了 V137 起加入的一项部署相关 observation 语义，而 simcode5 当前
明确强制关闭它：

```text
首次 confirmed hit 前: obs[49] = 0
首次 confirmed hit 后: obs[49] = clip(
    (control_time - represented_ball_state_time) / 0.5, 0, 1)
```

球位置、球速度、`ball-racket` 和 `obs[49]` 必须来自同一个 represented
timestamp。若 predictor 已传播到当前 query time，represented timestamp 也应为
当前时刻；不能把旧 raw detection timestamp 附到已前推状态。训练覆盖首次击球后
额外 0/1/2 个 5 ms tick 的 ball-only latency，实物端使用真实队列/时间戳，不能再
随机注入第二层延迟。represented age 超过 350 ms 时必须 fail closed。

因此建议在实物仓库保留原 simcode5 不变，新增如 `gpu1_v153_qacc67` 的显式
profile，开启上述 age 语义并绑定本模型 SHA。完成逐字段 parity 后，才可以把这个
独立 profile 作为 simcode5 系列的新模型入口。

## 3. 67-D observation 与输出

顺序与 GPU1-QACC/V5 保持：

```text
q[7], dq[7], base_q[3], base_dq[3],
ball_pos[3], ball_vel[3], racket_pos[3], racket_vel[3],
ball-racket[3], previous_action[7], q_cmd-q[7], ball_age_norm[1],
tau_norm[1], dq_cmd[7], delayed_active_q-q[7],
contact_time[1], contact_time-0.045[1]
```

QACC 更新仍为：

```text
qacc  = clip(actor(obs), -1, 1) * acceleration_limit
dq_cmd = clip(dq_cmd + qacc * 0.005, -velocity_limit, +velocity_limit)
q_cmd  = clip(q_cmd + dq_cmd * 0.005, hard_q_low, hard_q_high)
```

速度上限为 `[210,210,240,240,300,300,300] deg/s`，加速度上限为
`[1300,1300,1800,3000,3000,3000,3000] deg/s²`。action filter、jerk filter、
anti-windup、compensation 和 planner 均保持关闭。critic 的 279-D 输入只用于训练，
不得进入实物 actor。

## 4. 仅训练侧的变化

以下变化影响策略学到的权重或训练分布，但不是实物运行参数：

| 项目 | GPU1-QACC/V5 | V153 |
| --- | --- | --- |
| 仿真底座 | legacy floating | `aligned_fixed` |
| nominal 球质量 | 旧球合同 | `0.004 kg` |
| 球质量 DR | `[2.45,2.95] g` | `[3.9,4.1] g` |
| reset | racket launch | falling-contact B2→B3 homotopy；所选 stage 为 25% bridge |
| critic | 231-D | 279-D privileged/history；actor 仍 67-D |
| actuator J6 delay | 55 ms | 50 ms |
| 双域 gate | 无 | recovery + unchanged main retention 两套冻结 validator |

V153 相对直接父系 V137 的 actor inference contract 和执行器标定没有再变；主要
变化是 fixed-base/4 g/falling-contact reset、remaining-horizon penalty 和双域
curriculum/gate。所选 checkpoint 停在 25% B2→B3 bridge，不应写成已通过最终
100% B3 或最终 main-task proof。

训练 actuator 逐关节 nominal 值为：

```text
delay_ms = [45,50,45,40,35,45,50]
wn_rad_s = [21.89113794,22.08957538,22.65047050,21.73053300,
            20.15494836,22.24513809,23.05467182]
zeta     = [0.3330028,0.3295521,0.3111642,0.3114000,
            0.3131064,0.3112326,0.2855348]
```

这些值和 `sport_taskspace_fit_v1` XML PD 都是训练侧真实 drive surrogate；不得
把 delay/filter/PD 再叠加到实物策略输出。实物仍发布 current `q_cmd`。

## 5. 训练代码与复现入口

V153 的精确 source snapshot 与当前工作文件哈希一致。训练 companion 批次应至少
包含：

```text
TRAINING_CURRICULA.md
pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py
pingpong_controller/tools/rl_sim/train_juggle_mjx_ppo.py
pingpong_controller/tools/rl_sim/mjx_juggle_env.py
pingpong_controller/tools/rl_sim/mjx_smoke.py
pingpong_controller/tools/rl_sim/ball_mass_measurement.py
pingpong_controller/tools/rl_sim/run_with_host_memory_guard.sh
pingpong_controller/tools/rl_sim/moz1_pd.xml
pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v153_dual_domain_from_v152_stage51.sh
pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v153_resume1_stage57_u1.sh
test/test_fixed_base_ball4g.py
test/test_goal_d455_curriculum.py
```

所选 checkpoint 来自 `resume1_stage57_u1` launcher 的续训目录；初始 launcher
描述 V152 Stage-51 到 V153 的入口，resume1 launcher 固定了中断点、optimizer、
curriculum history、actor anchor/replay、双域证据和 W&B step offset。它们是训练
审计材料，不是实物启动脚本。

## 6. 实物分阶段门槛

依次完成：模型哈希/维度/golden-vector、67-D observation 逐字段 parity、固定输入
action/`q_cmd` parity、recorded replay、shadow target、无球低增益限空间、监督有球。
必须记录 actor observation、raw action、限幅后 QACC、`dq_cmd/q_cmd`、represented
ball timestamp/age、confirmed-hit 状态、safety clamp 和停止原因。任何 joint/ball
时间戳失效、NaN/Inf、维度或模型身份不匹配都必须拒绝推理或进入受监督安全状态。
