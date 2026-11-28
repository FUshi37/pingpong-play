# V325 wind-early r5_fast u6247 实物部署

```bash
PROBE_CONFIG_ONLY=true POLICY_RELEASE_ID=fplc_gpu0_v325_wind_early_r5fast_u6247 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
POLICY_RELEASE_ID=fplc_gpu0_v325_wind_early_r5fast_u6247 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

第一条命令只检查配置。启动后可在网页 RL profile 选择 **V325 wind-early r5_fast u6247 (120D，固定±5°)**。切回 u416 时选择原网页选项或使用 `POLICY_RELEASE_ID=fplc_gpu0_v325_u416` 启动。切换后重新使能发布。

| 项目 | u6247 部署值 |
| --- | --- |
| 来源 | `simcode12/pingpong_moz1-wind-early` 的 u6247 发布快照 |
| 权重 SHA-256 | `e9591ce1eabb2100be31a4a73e58c8a1b2c4bc97c3100edfa5fdf49698ab9666` |
| Actor | 单个 120→282→282→282→7 MLP，tanh 隐层，确定性均值输出 |
| 时钟 | 控制 200 Hz，球观测默认 90 Hz，策略年龄按缓存刷新时钟计算 |
| 风观测 | 500 ms 实测历史的因果风估计。第 5 次合格起球后启用时间特征。强风特征在估计风速超过 1.25 m/s 后渐增 |
| 姿态 | 内层 roll/pitch 从复位起固定 ±5°。实物外层保护沿用 ±7°，yaw 自由 |
| 内层工作空间 | X [−0.348, 0.348]、Y [−0.700, −0.280]、Z [1.000, 1.150] m |
| 实物外层工作空间 | X [−0.398, 0.398]、Y [−0.750, −0.230]、Z [0.950, 1.200] m |
| 指令 | 7 轴归一化速度经原生约束投影积分一次，持续维护 `q_ref` 并下发机器人 RMP |

训练范围包含最高 1.6 m/s 的恒风和动态风。训练的随机起风时刻属于环境状态。实物和网页 CPU 镜像都用实际输入的球位置历史构造 actor 特征。第 5 次合格起球在实物由球拍附近的球运动估计，原始测量年龄继续用于有效性检查及记录。

u6247 的 pickle 省略了物理配置。训练的 `migrated_stack.py` 逐项校验物理配置与 u416 清单一致，因此部署从 **simcode12 内**的配套 u416 清单恢复物理配置。启动校验固定 u6247 权重、XML、模型合同和两份清单的哈希。发布文件及录制均可追溯到独立的 u6247 标识。`simcode12/COLCON_IGNORE` 使训练目录退出 colcon 包发现。

随模型发布的冻结评估覆盖无风、早期恒风、早期动态风、晚期恒风和晚期动态风，60 s 严格通过率为 1280/1280。实物表现和完整 1200 s 验收待验证。
