# V325 wind-early Sim2Real S09 u7447 部署

```bash
PROBE_CONFIG_ONLY=true POLICY_RELEASE_ID=fplc_gpu0_v325_wind_early_s2r_u7447 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
POLICY_RELEASE_ID=fplc_gpu0_v325_wind_early_s2r_u7447 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

网页 RL profile 可选择 **V325 wind-early Sim2Real S09 u7447 (120D，固定±5°)**。切回上一版时选择 u6247，或使用 `POLICY_RELEASE_ID=fplc_gpu0_v325_wind_early_r5fast_u6247` 启动。

| 项目 | 部署值 |
| --- | --- |
| 训练版本 | `simcode12` 中的 S09 `D2_dr65`，u7447，稳态初始化权重 1.0，域随机化强度 0.65 |
| 权重 SHA-256 | `a40da56cefb1cc6a8928a8199667cf1379a0bbdac9326949befd15a3bc089bcd` |
| 网络与时钟 | 单个 120→282→282→282→7 MLP。控制 200 Hz，球状态默认 90 Hz，支持 120 和 200 Hz |
| 球观测 | 缓存刷新年龄、实测球历史和因果风估计。原始测量年龄用于有效性检查和录制 |
| 指令 | 归一化关节速度经训练投影积分一次，持续维护 `q_ref` 并下发机器人现有 RMP |
| roll/pitch | 训练投影固定 ±5°，实物外层保护沿用 ±7° |
| 实物外层工作空间 | X [−0.398, 0.398]、Y [−0.750, −0.230]、Z [0.950, 1.200] m |
| CPU 镜像 | 每次复位以冻结源码中的 `P_warm` 初始化七轴目标估计器，镜像状态记录此矩阵 |

训练中的重力倾斜、动捕延迟、静态偏置、噪声和碰撞参数扰动都在仿真里生成。实物使用当前校准下的传感器输入和真实机器人 RMP。网页 CPU 镜像随实测球输入，`P_warm` 用于镜像内的 RMP。

启动校验固定 u7447 权重、XML、模型合同与两份清单的哈希。运行时核对 checkpoint 的训练计划和课程阶段，再从 `simcode12` 内的基准物理清单恢复名义配置。`simcode12/COLCON_IGNORE` 排除训练树。

随模型附带的 60 秒冻结评估中，`P_warm` 严格通过 **1280/1280**，实物孪生仿真严格通过 **640/640**。`P_cold` 的物理存活为 **1280/1280**，严格通过率为 **97.19%**。实物试验和完整 1200 秒验收待验证。
