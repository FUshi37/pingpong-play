# Drag Ball 实物指令 Replay 与 RMP 参数拟合

录制：`probe_recording_20260829_180205`

## Replay 等价性门槛

- 记录的 sim q_ref 离线重放 ↔ 在线 sim mirror：q RMS=0.000000437 rad，dq RMS=0.0000621 rad/s，`passed=true`。
- 本工具实物 q_ref baseline replay ↔ 先前 common replay：q RMS=0.000000000 rad，dq RMS=0.0000000 rad/s，`passed=true`。

因此离线 replay 与当前 sim 的模型、observe/apply 相位和 RMP 状态机一致。实物 q_ref replay 不应与在线 independent sim 后半段完全相同，因为两边的 actor q_ref 从 0.085 s 起已经因 q/dq 状态分叉而不同。

## 拟合结果

| 窗口 | baseline q RMS | tuned q RMS | q 改善 | baseline dq RMS | tuned dq RMS | dq 改善 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全程 | 0.005004 | 0.003523 | 29.6% | 0.11829 | 0.06580 | 44.4% |
| 1 s 后 | 0.004238 | 0.002406 | 43.2% | 0.11181 | 0.05262 | 52.9% |
| 训练 1–9 s | 0.004691 | 0.002760 | 41.2% | 0.12071 | 0.05869 | 51.4% |
| 验证 9–14 s | 0.003463 | 0.001752 | 49.4% | 0.09711 | 0.03851 | 60.3% |
| 留出测试 14 s–结束 | 0.004170 | 0.002358 | 43.5% | 0.11056 | 0.05482 | 50.4% |

搜索共运行 89 个唯一候选。参数选择使用 1–9 s 训练段和 9–14 s 验证段；14 s 之后的结果在搜索完成前未参与选择。

## 主频幅相诊断

| 关节 | q_ref 主频 Hz | baseline 实物/sim 幅值比 | tuned 实物/sim 幅值比 | baseline sim-real 相位 | tuned sim-real 相位 |
| --- | ---: | ---: | ---: | ---: | ---: |
| J1 | 4.269 | 1.0092 | 1.0092 | -4.33° | 0.30° |
| J2 | 1.038 | 1.0001 | 1.0009 | -2.17° | -0.33° |
| J3 | 4.269 | 0.9943 | 0.9945 | -9.56° | 1.22° |
| J4 | 4.269 | 1.0097 | 1.0097 | -4.51° | 0.11° |
| J5 | 4.269 | 1.0217 | 1.0218 | -6.02° | 0.14° |
| J6 | 4.269 | 1.0279 | 1.0280 | -5.93° | 0.23° |
| J7 | 4.269 | 1.0254 | 1.0255 | -5.78° | 0.38° |

幅值比调参前后几乎不变并已接近 1，说明无需修改 kp/kd。相位则从 baseline 的约 -2° 至 -10° 收敛到大约 ±1.2° 内；拟合本质是逐关节时序修正。

## 选中的相对参数

```json
{
  "kp_scale": 1.0,
  "kd_scale": 1.0,
  "output_filter_length": 35,
  "target_filter_length": 10,
  "target_update_delay_steps": 4,
  "output_delay_offset_steps": -5,
  "output_delay_per_joint_adjustment": [
    1,
    -1,
    -3,
    1,
    0,
    0,
    0
  ],
  "joint_velocity_feedforward": 0.5,
  "acceleration_limit_scale": 1.0
}
```

对应 checkpoint `env_cfg` 建议值：

```json
{
  "recovered_rmp_jnt_kp_scale_per_joint": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "recovered_rmp_jnt_kd_scale_per_joint": [
    1.0,
    1.0,
    0.92,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "recovered_rmp_output_filter_length": 35,
  "recovered_rmp_target_filter_length": 10,
  "recovered_rmp_target_update_delay_steps": 4,
  "recovered_rmp_output_delay_steps_per_joint": [
    14,
    13,
    11,
    14,
    13,
    13,
    13
  ],
  "recovered_rmp_joint_velocity_feedforward": 0.5,
  "arm_acc_limit_deg_s2": [
    1300.0,
    1300.0,
    1800.0,
    3000.0,
    3000.0,
    3000.0,
    3000.0
  ]
}
```

## 结论与使用边界

- 同一 q_ref 下，实物和 baseline sim 的主要受迫频率本来就一致；肉眼看到的在线频率差主要来自闭环 actor 在状态分叉后输出了不同 q_ref，而不是两个执行器自由运行在不同频率。
- 原 output delay 为 [18,19,19,18,18,18,18] 个 1 ms step、target_update_delay_steps=3；建议 delay 为 [14,13,11,14,13,13,13]、target_update_delay_steps=4。合并计算后，各关节净提前约 [3,5,7,3,4,4,4] ms。
- 本次拟合只改变离线 recovered-RMP/MuJoCo replay，没有改 checkpoint、实时 mirror 或实物控制。
- 单条 drag-ball 轨迹不足以直接发布新 plant。只有留出段同样改善，并且另一条独立录制也验证后，才建议把这些值写入下一版训练/部署 checkpoint；否则容易把 RMP 启动隐藏状态或本次负载拟合进去。
- `01–02` 为各关节调参前后曲线，`03` 为逐 tick gap，`04` 为主频幅相比较。
