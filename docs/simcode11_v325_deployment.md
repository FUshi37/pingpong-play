# V325 交互探针适配

启动 V325：

```bash
POLICY_RELEASE_ID=fplc_gpu0_v325 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

只校验模型、XML 和配置，不初始化 ROS、相机或机器人：

```bash
PROBE_CONFIG_ONLY=true POLICY_RELEASE_ID=fplc_gpu0_v325 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

网页的 RL profile 可选择 `V325 S15 u14400 恒风扰动 1.6 m/s (114D)`。
默认启动版本仍为 V172；选择 V325 后使用独立的 checkpoint 和观测契约。

模型为 `converged_s15_u00014400.pkl`，SHA-256 为
`fd9def8f5fdb511e1543cbcca264e0021738ff07c6425e8d393cb1c240918e97`。
S15 的训练风速上限为 1.6 m/s，actor/critic/action 维度为 114/453/7。
训练记录表明 S15 已收敛；checkpoint 的独立评估字段为空，最终验收字段为 false。

114D actor 输入由原有 72D 位置观测加 42D 后缀组成：

- 0/50/100/150/200/300/400/500 ms 实测球位置相对当前实测位置的差，共 24D。
- 8D 真实采样年龄（除以 0.6 s）和 8D 有效标记；尚无历史的槽位无效。
- 2D 当前 roll/pitch 权限（除以 20°）。

历史每 5 ms 更新，后缀始终使用直接传感器观测及其真实年龄。预测位置不会替代后缀中的
实测球位置。reset、干预或模型重新进入时清除历史。动作按训练源码合成 teacher、
有界 residual 和姿态权限门控，之后进入既有 Cartesian projection 与 persistent q_ref。
运行时不依赖训练 JAX 模块，也不生成额外的风力或传感器噪声。

该 checkpoint 的课程目标为 20°，但训练 `effective_degrees` 在每次 reset 后先使用
2°：确认 5 次合格起球，再按 `sin²(πt/2)` 在 1 秒内从 2° 放开到 20°。
114D 最后两维、残差门控、线性投影和非线性检查同步使用当前权限。
V325 实机 roll/pitch 外层停机边界仍为
各 ±20°，相对训练参考姿态计算，yaw 不受这两个角度边界限制。实测姿态和命令目标均检查
此边界。其他 FPLC 模型继续使用原有各 ±4° 的实机边界；位置边界与反馈时效要求保持生效。

V325 策略内部的 Cartesian projection 保留冻结训练边界；实机外层安全工作空间
在训练边界的六个方向各扩展 5 cm，用于容纳执行响应相对命令参考的偏差：

|轴|训练投影范围 / m|实机外层安全范围 / m|
|---|---|---|
|X|[−0.348, +0.348]|[−0.398, +0.398]|
|Y|[−0.700, −0.280]|[−0.750, −0.230]|
|Z|[1.000, 1.150]|[0.950, 1.200]|

范围对应训练使用的拍面 site、策略笛卡尔坐标系。独立模型契约的
`real_workspace_mode=training` 从 `env_contract` 读取训练边界，
`real_workspace_margin_m=0.05` 派生实机外层范围；V325 不再与
通用 `config_done_monitoring.yaml` 或 `DONE_EE_*` 的位置框求交，因此默认配置
不会再截断训练范围。直接启动、网页切换、复位检查、实测/目标命令检查、done monitor
和网页 EE 框使用相同边界。切换回其他模型时恢复其原来的位置配置。
初次工作空间对齐的备份保存在 `recovery_artifacts/v325_workspace_alignment_20261120/`；
本次外扩 5 cm 的独立备份、diff、验证与回退入口为
`recovery_artifacts/v325_workspace_margin_5cm_20261120/`。
这项调整不改变模型权重、训练投影、触球模型、观测时延或姿态放权公式。

实机没有仿真接触标记，使用直接传感器球位/球速和实测拍面 FK 估计起球确认。
球接近拍面后，须达到训练的离拍高度、预测顶点、确认时限和最短计数间隔才计数，
而非直接采用相机冲击数或录制里的低矮反弹数。该传感器估计支持动捕和相机；
它与仿真的精确接触判定仍有差别。干预/reset 清除计数和放权时钟。
录制 `fplcProjection.windStartup` 保存当前计数、权限、残差 gate 和确认方法。
此次修复的备份、差异和回退脚本位于
`recovery_artifacts/v325_startup_authority_fix_20261119/`。

此次姿态边界调整有独立的修改前备份和回退入口：
`recovery_artifacts/v325_adaptation_20261118/real_attitude_20deg/`。

控制频率保持 200 Hz，球状态刷新支持 90/120/200 Hz。`simcode11/COLCON_IGNORE`
使 colcon 跳过训练快照；模型通过文件路径加载。

本次未迁移 simcode1–10。所有既有发布条目和模型/XML 路径保持原值。
适配备份、文件校验及回退入口位于 `recovery_artifacts/v325_adaptation_20261118/`。
