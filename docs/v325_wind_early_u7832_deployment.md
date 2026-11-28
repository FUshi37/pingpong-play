# V325 wind-early Sim2Real r6 W2.8 u7832 部署

```bash
PROBE_CONFIG_ONLY=true POLICY_RELEASE_ID=fplc_gpu0_v325_wind_early_s2r_r6_u7832 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
POLICY_RELEASE_ID=fplc_gpu0_v325_wind_early_s2r_r6_u7832 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

网页 RL profile 选择 **V325 wind-early Sim2Real r6 W2.8 u7832 (120D，固定±15°)**。切回上一版时选择 u7447，或使用 `POLICY_RELEASE_ID=fplc_gpu0_v325_wind_early_s2r_u7447` 启动。

| 项目 | u7832 部署值 |
| --- | --- |
| 权重 SHA-256 | `cdb2967ec4b3706502d7417d27b11925709d216e49c746fd7c3922f72efdf99a` |
| 网络与时钟 | 单个 120→282→282→282→7 MLP。控制 200 Hz，球观测默认 90 Hz |
| 观测 114–115 | 因果估计风速 XY，模长在 4.5 m/s 限幅，除以 4.5 后输入 |
| 观测 118–119 | 先按 4.5 m/s 恢复模长，再应用 1.25 m/s 死区与 0.35 m/s 过渡宽度 |
| 起球与姿态 | 第 5 次合格起球后启动时间特征。投影从复位起固定 ±15° roll/pitch，内层保留 0.1° 余量，实物外层保护 ±17°，yaw 自由 |
| 工作空间 | 投影 X [−0.348, 0.348]、Y [−0.700, −0.280]、Z [1.000, 1.150] m。实物外层各向留 5 cm |
| RMP 与观测时钟 | 真实测量驱动策略，观测年龄使用缓存刷新时钟。下发持续积分的 `q_ref`。CPU 镜像使用 `P_warm` 并随实测球输入 |

启动校验使用独立合同和冻结清单，模型文件在 `simcode12`。网页与离线录制解析按实际 profile 标记风速归一化的单位。同段录制混用两种尺度时，回放保留各帧的 profile。`simcode12/COLCON_IGNORE` 将训练树排除在 colcon 包发现之外。

W2.8 表示 checkpoint 达到 2.8 m/s 训练课程阶段。4.5 m/s 是策略风速观测的量程。随附的 18 组 60 秒仿真严格通过 460/576。其中 RealExact 无风及 +X 1.6、2.4、2.8 m/s 各通过 32/32。RealExact −X 2.8 m/s 通过 22/32，两组 RealExact 动态风均为 0/32。实物试验待验证。
