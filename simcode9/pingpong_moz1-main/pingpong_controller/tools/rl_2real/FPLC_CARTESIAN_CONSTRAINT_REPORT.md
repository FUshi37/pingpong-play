# FPLC 笛卡尔约束提交报告

2026-10-10。对应 FPLC GPU0 V152／V161、GPU1 V151／V172 的训练与 validate 路径。

## 提交实现

`tools/rl_sim/cartesian_pose_action_projection_jax.py` 提供训练环境使用的 JAX 位姿约束投影；`cartesian_pose_action_projection.py` 提供 NumPy 参考实现，供实物移植与离线逐步配对。两者不引入另一层 RMP。

环境中的集成、FK／Jacobian、非线性回退以及 previous_action 更新随第一批后续仿真提交保存，不能仅提交这两个求解模块就认为模型运行依赖已经完整。

## 必须保持的模型合同

四个模型均使用 200 Hz、7-D QVEL 和持续 q_ref。关节速度上限为 `[210,210,240,240,300,300,300]` deg/s，内部转换到 rad/s；动作比例 1，动作滤波时间常数 0。保存配置中的逐关节 reference error horizon 均为 **10 秒**，不是早期 bounded-reference 的约 18 ms。

模型注册 policy frame 中的位置下界为 `[-0.348,-0.700,1.000] m`，上界为 `[0.348,-0.280,1.150] m`；相对参考姿态 roll／pitch 各 2°，yaw 不启用该约束。位置／姿态裕量为 0.5 mm／0.1°；两类 CBF gain 配置值为 20；Dykstra 扫描 24 次，tolerance `2e-5`。字段数值与模型配置 JSON 配对，实物必须核对坐标变换和参考姿态。

先用持续 reference 的 FK／Jacobian 和 measured q 约束可行速度，再积分并执行环境规定的非线性回退。投影后归一化动作回填 previous_action。不得改为每 tick 从 measured q 重新施加一个 stateless lead，也不得省略回退或仅检查向量维数。

## 实物修改指导与验证状态

在独立实物仓库的 `pingpong_node.py` 和策略接口中增加版本化 FPLC q_ref／projection 状态、reset 与日志；按 `FPLC_REAL_ROBOT_MIGRATION_REPORT.md` 逐字段和多 tick 配对。保持 policy q_ref → 既有安全层 → `MechUnitCmd.jnt_pos` → 机器人已有 RMP 的单一边界。

本次仅整理已有实现与提交范围，未改变控制行为，未运行训练或硬件试验。四个模型 SHA 已复核；此前通过的 CPU 课程／恢复检查仍只属于仿真接口证据。按用户要求，本次不提交 `test/` 或其他回归测试代码。
