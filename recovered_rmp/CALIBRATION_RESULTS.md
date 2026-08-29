# record_new3/4 的 RMP/MJX 标定结果

## 当前状态（2026-08-13）：RMP 已恢复；实物伺服 1:1 尚未达到

本轮把 DataTracer 中同一段实物执行的最终 RMP reference 和编码器反馈分开评估。对每关节
允许一个共同的连续下游延迟（5--25 ms，步长 0.05 ms），仍得到：

| 比较层 | q RMS | dq RMS | ddq RMS |
|---|---:|---:|---:|
| 最终 RMP reference（最优延迟）vs 编码器 feedback | 0.156158° | 0.078217 rad/s | 10.200634 rad/s² |
| 完整 MJX、3 ms 指令接收 + 逐关节输出延迟 + 高带宽 PD（`210018` 留出） | 0.235242° | 0.064755 rad/s | 2.860556 rad/s² |

所以当前**没有**任何可写为训练默认值的参数组合能达到 `0.1° / 0.05 rad/s / 1 rad/s²`。
第一行尤其重要：它没有 MuJoCo 或 PD 误差，直接表明实物 RMP reference 到 encoder feedback
之间存在未记录的伺服、机构或反馈动态；仅调 RMP/PD 不能把这部分加速度残差消掉。下游
延迟的识别结果约为 `12.3--14.6 ms`，而上游应用目标到 PlanTask 的接收延迟为约 `3 ms`，二者
已经分别接入恢复链路。

`validate_rmp_datatracer.py --include-feedback-reference-baseline` 是该结论的可复现实证。它的
ddq 使用 1 kHz `actual_qd` 的前向差分，和 record replay 的指标口径一致；不能用平滑后的
central difference 代替。

也对 `1.410--3.200 s` 拟合、并对 `3.200--5.000 s` 无 teacher-forcing 开环预测的因果
ARX 伺服模型做了穷举阶数/滞后扫描。留出总体为 `0.233927° / 0.108167 rad/s /
10.470737 rad/s²`；即使 J0/J1/J3/J4/J5/J6 的速度可以落到 `0.013--0.024 rad/s`，其
加速度仍为 `7.03--15.53 rad/s²`，J2 的速度本身也不可预测。结果保存在
[`datatracer_203806_servo_arx_holdout.json`](calibration/datatracer_203806_servo_arx_holdout.json)，
并可由 [`identify_datatracer_servo_arx.py`](identify_datatracer_servo_arx.py) 重跑。这排除了
“缺一次普通 PD 或线性低通扫描”的解释。

要兑现三个硬阈值，需要在同一动态轨迹上重复采集最终 RMP q/qd/qdd、编码器 q/qd、实际
电流/力矩以及低层模式/反馈滤波状态，并允许把辨识出的闭环伺服模型并入仿真。用户明确要求
“不启用执行器模型”的路径不能独自解释这段误差，故当前保持无执行器模型路径而不伪造通过。

## 2026-08-12：1 kHz DataTracer 重新分层验证

旧标定把“上游 200 Hz 位置目标 → RMP 输出”和“RMP 输出 → 实物反馈”混在一起，因为
record_new3/4 当时没有显式保存最终 `axis_pos/axis_vel/axis_acc`。现已从同次运行的 Movax
DataTracer 取得这三路 1 kHz 参考，并确认此前漏了目标变化限速、事件驱动 Kalman 和
`filter_length=35` 输出均值。

恢复 YAML 原值、不启用 jerk 限制，在 `record_new4/203833` 对应 DataTracer 的稳定区间
`1.410–5.000 s`（3590 点）得到：

| 比较层 | q RMS | dq RMS | ddq RMS |
|---|---:|---:|---:|
| 恢复 RMP 输出 vs 实机 axis reference | **0.038381°** | **0.010817 rad/s** | **0.303445 rad/s²** |
| 理想 MuJoCo PD 输出 vs 实机 feedback（旧稳定 PD） | 1.242° | 0.2320 rad/s | 13.084 rad/s² |
| 理想 MuJoCo PD 输出 vs 实机 feedback（7 关节联合扫描） | 0.734° | 0.1978 rad/s | 11.694 rad/s² |

第一行只证明恢复的 **RMP 输出层** 已通过 `0.1° / 0.05 / 1`；它不是实物 encoder-feedback
的 1:1 证明。后两行以及上面的新反馈基线都没有通过。仅调理想刚性 PD 不能解释实物低层的
摩擦、柔顺、间隙和测量链，因此未把联合扫描参数设为默认。下面旧的
`record_new3_4_q_qd_qdd` 数值保留用于版本对照，不应再被描述成推荐的原厂 RMP 参数。

## 控制链和指标口径

所有结果只在记录中 `policyExecuted=true` 的样本上计算，位置/速度单位为 SI。ddq 与
项目正式 sim-real 报告使用完全相同的测量算子：实物速度按 `ros.jointSampleStampS`、
仿真速度按 `sourceTimeS` 做当前 tick 到下一连续 `sourceSeq` 的前向差分；重复采样、
序号缺口和末行不参与 RMS。MJX 瞬时 `data.qacc` 只保留为诊断量。完整 MJX
回放链为：

```text
recorded command.position
  -> recovered joint RMP (1 ms, 每个记录目标保持 5 个周期)
  -> q/qd/qdd
  -> XML low-level PD (1 ms)
  -> MJX physics (1 ms)
```

`actuator_cmd_filter`、FOPDT、command delay、inverse compensation、额外 servo planner 和
actual-state projection 均关闭。XML position actuator只承担低层 PD，不是执行器动态模型。

下面两个配置必须区分：

| profile | RMP Kp/Kd | XML PD Kp/Kd scale | qdd FF | 含义 |
|---|---:|---:|---:|---|
| `recovered_yaml_pd_fit` | `400 / 40` | `0.02 / 0.10` | 0 | RMP 使用二进制/YAML 恢复值；只拟合无法从日志直接得到的低层 PD |
| `record_new3_4` | `120 / 21.9089` | `0.03 / 0.10` | 0 | RMP 与 PD 的跨记录经验拟合；不是原厂 RMP 参数 |
| `record_new3_4_q_qd_qdd_cpu` | `151.198 / 13.4403` | 7组独立值 | 0 | 独立 MuJoCo 理想力矩模型联合拟合 q/dq/ddq |
| `record_new3_4_q_qd_qdd` | 7组独立值 | 7组独立值 | 0 | 完整 MJX 联合拟合 RMP/PD/damping/armature；训练推荐值 |

未缩放的 XML PD 配合当前 MuJoCo 浮动全身模型会数值不稳定，所以不能把 XML 原值误称为
硬件低层增益。硬件低层控制器的私有增益并不在 `mc_core` RMP YAML 中。

## 独立 MuJoCo 理想力矩 PD 回放

| 记录 | policy 样本 | 恢复 RMP + PD fit | `record_new3_4` fit |
|---|---:|---:|---:|
| `record_new3/195203` | 350 | 0.046185 rad / 2.646° | 0.030714 rad / 1.760° |
| `record_new4/203653` | 998 | 0.038680 rad / 2.216° | 0.028433 rad / 1.629° |
| `record_new4/210018` | 441 | 0.047253 rad / 2.707° | 0.035136 rad / 2.013° |

## 完整 MjxJuggleEnv 回放

| 记录 | profile | q RMS | dq RMS |
|---|---|---:|---:|
| `record_new3/195203` | `recovered_yaml_pd_fit` | 0.046717 rad / 2.677° | 0.7382 rad/s |
| `record_new3/195203` | `record_new3_4` | 0.030364 rad / 1.740° | 0.4995 rad/s |
| `record_new4/203653` | `record_new3_4` | 0.026167 rad / 1.499° | 0.4494 rad/s |
| `record_new4/210018` | `record_new3_4` | 0.034335 rad / 1.967° | 0.5010 rad/s |

联合 q/dq/ddq 标定后的结果如下；前两段用于拟合，`210018` 完全留出：

| 记录 | q RMS | dq RMS | ddq RMS |
|---|---:|---:|---:|
| `record_new3/195203` | 0.01767 rad / 1.012° | 0.1619 rad/s | 4.43 rad/s² |
| `record_new4/203653` | 0.01962 rad / 1.124° | 0.2111 rad/s | 5.48 rad/s² |
| `record_new4/210018`（留出） | 0.01861 rad / 1.066° | 0.2026 rad/s | 5.06 rad/s² |

与上一版 `record_new3_4_q_qd_qdd` 在正确 ddq 口径下的留出结果相比，角度从约 1.18°
降到 1.07°，速度从 0.258 降到 0.203 rad/s，加速度从 6.61 降到 5.06 rad/s²。三路
同时改善，而不是用更差的速度或加速度换取位置曲线。

推荐 profile 的 XML 有效 PD（不是缩放比例）为：

| 关节 | Kp | Kd |
|---:|---:|---:|
| J0 | 784.291 | 27.245 |
| J1 | 583.191 | 18.007 |
| J2 | 868.644 | 9.500 |
| J3 | 550.635 | 12.978 |
| J4 | 153.140 | 1.013 |
| J5 | 413.879 | 2.024 |
| J6 | 134.876 | 0.686 |

自动标定分为两层：[`calibrate_rmp_pd.py`](../simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/calibrate_rmp_pd.py)
先在独立 MuJoCo 中联合搜索 RMP 与7组 PD；随后
[`calibrate_rmp_pd_mjx.py`](../simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/calibrate_rmp_pd_mjx.py)
把不同 RMP/PD/damping/armature 候选作为并行 MJX 环境，以 `qpos/qvel` 和同口径速度
差分 ddq 做逐关节精调。联合损失默认
权重为 q/dq/ddq=`0.4/0.3/0.3`，分别按 `0.02 rad / 0.25 rad/s / 5 rad/s²` 归一化。

独立回放和环境内回放在 `record_new3/195203` 上分别为 1.760° 与 1.740°，说明把 RMP
接入 5 个 MJX 子步没有额外引入明显零阶保持或状态差分误差。经验 profile 在两段
`record_new4` 上仍得到 1.50° 和 1.97°，比恢复 RMP 基线更好，具有有限的跨记录泛化，
但还不能称为 1:1。

因果扫描了 0/5/10/15/20 ms 的 RMP 输入延迟。训练集偏好 5 ms，但它使完全留出的
`210018` 的 q/qd/ddq 三项全部变差，所以推荐 profile 明确保留 0 ms；没有把评估时移
伪装成物理或执行器延迟。

## 可复现命令

独立 CPU replay：

```bash
python simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/replay_rmp_recording.py \
  <control_samples.jsonl> \
  --calibration-profile record_new3_4 \
  --output /tmp/rmp_replay
```

完整 MJX replay：

```bash
cd simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim
python replay_rmp_recording_mjx.py <control_samples.jsonl> \
  --calibration-profile record_new3_4_q_qd_qdd \
  --output /tmp/rmp_mjx_replay
```

重新执行两级标定：

```bash
python calibrate_rmp_pd.py --output /tmp/rmp_pd_cpu.json

JAX_PLATFORMS=cpu python calibrate_rmp_pd_mjx.py \
  --output /tmp/rmp_pd_mjx.json
```

进一步扫描时可独立覆盖 `--rmp-jnt-kp-scale`、`--rmp-jnt-kd-scale`、
`--controller-kp-scale`、`--controller-kd-scale` 和
`--acceleration-feedforward-s2`。

## 达到真正 1:1 仍缺少的数据

现有记录保存了 RMP 的输入位置和机器人实测 q/dq/ddq，却没有保存 `mc_core` 最终
`axis_pos/axis_vel/axis_acc`。因此 RMP 内部轨迹误差和低层伺服/机械误差目前不可唯一分离。
下一轮实物采集至少应同步记录：

1. 1 kHz RMP 最终 q/qd/qdd；
2. 控制器实际接收时间戳和编码器采样时间戳；
3. 关节实际力矩/电流、低层 PD 与 qdd 前馈模式；
4. 同一条轨迹多次重复，用于区分确定性模型偏差和摩擦/间隙噪声。

拿到这些量以后，才有依据继续辨识低层控制器、摩擦、柔顺和时间偏移，并验证“1:1”而
不是只把位置曲线拟合得更接近。
