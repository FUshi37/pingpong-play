# FPLC 仿真策略与课程提交报告

2026 年 10 月 10 日。适用 FPLC GPU0 V152 与 GPU1 V151，以及沿相同控制和 actor 合同推进的 clean 后继 V161、V172。本次归档策略、物理和课程实现；实物迁移通过新接口完成，保留旧 57-D QVEL 与 67-D QACC 路径。

## 共用代码与版本关系

`tools/rl_sim/mjx_juggle_env.py` 实现观测历史、固定基座和 4 g opt-in 物理、物理接触／确认击球、奖励与终止、测量反馈 QVEL、持续 q_ref 和位姿投影。`fixed_contact_plane_training.py` 提供接触预测及目标几何；`fixed_plane_v92.py`、`fixed_plane_v93.py`、`fixed_plane_v99.py`、`fixed_plane_v149.py` 定义两类 actor 输入。`cartesian_pose_action_projection_jax.py` 和 NumPy 参考 `cartesian_pose_action_projection.py` 保存投影方法，`recovered_rmp_jax.py` 与 `delay_control.py` 保存仿真执行路径。

`train_juggle_mjx_curriculum.py`、`train_juggle_mjx_ppo.py`、`fixed_plane_v151.py`、`fixed_plane_v152.py` 及所选四模型需要的后继模块保存课程、PPO 和继续训练合同，随第一批训练／validate 提交保存；根目录 `TRAINING_CURRICULA.md` 同批登记。V153／V154 整理 DR 桥与质量阶段；V155 插入精度桥；V156／V158 增强运动和触球质量；V159／V161 使用均值质量目标；V172 对齐质量门。V157 仍是 V158 的直接 import 依赖，保留该模块不表示提交 V157 模型或历史实验包。V168／V169 等独立实验文件不在本次范围。

当前训练器／环境整文件中也包含其他 opt-in 分支；本次不为这些未选定分支补齐独立模块和历史运行包，不宣称本次是整个实验工作区的完整备份。四个选定课程和 validate 的依赖单独核对。物理 GPU 编号或文件名不能代替 checkpoint 元数据。

环境在启用相应奖励时延迟导入 V158／V159；课程恢复也延迟导入训练器。第一批依次保存约束求解、仿真／碰撞和训练／validate，三次全部完成后构成所选四模型的共同代码路径。带较早版本号的辅助模块只因实际 import 或启用奖励需要保留，不提交其旧模型、启动脚本或实验记录。

## 实物需要复现的策略合同

四个模型均为 7-D QVEL，控制周期 5 ms；仿真 recovered RMP 与 MJX 为 1 ms。输出经过关节速度缩放、持续 q_ref 下的笛卡尔位姿投影和非线性回退，再作为位置目标进入 RMP。投影后的归一化动作回填 previous_action。历史每个 200 Hz tick 推进一次，90 Hz 相机无新帧时重复持有位置，同时更新样本年龄。

四个保存配置的 `recovered_rmp_qvel_reference_error_horizon_s_per_joint` 都是 `[10,10,10,10,10,10,10]` **秒**；不是旧 V14／V15 的约 18 ms。必须读取模型配置并验证该合同，不能套用旧 57-D adapter 的参数。速度上限为 `[210,210,240,240,300,300,300]` deg/s，内部先转 rad/s；动作比例为 1，动作滤波时间常数为 0。

位姿投影在其注册 policy frame 中使用位置下界 `[-0.348,-0.700,1.000] m`、上界 `[0.348,-0.280,1.150] m`，相对参考姿态 roll／pitch 各 2°，yaw 不受这个姿态门约束。位置／姿态裕量为 0.5 mm／0.1°，两类 CBF gain 配置值为 20，Dykstra 扫描 24 次，tolerance 为 `2e-5`。实物 FK、坐标变换、参考姿态和回退必须先逐步配对，不可只复制界限数字。

硬件发布边界是策略 q_ref 经现有安全层后写入 `MechUnitCmd.jnt_pos`，由机器人已有 RMP 执行。recovered RMP、XML PD、模拟延时、MuJoCo solref／solimp 都留在仿真侧。完整观测索引和实物文件修改位置见 `FPLC_REAL_ROBOT_MIGRATION_REPORT.md`；碰撞版本见 `BALL_RACKET_CONTACT_REAL11_STATIC_1MS_V1_REPORT.md`。

## 验证与范围

本轮此前已通过 41 项 CPU 课程／恢复合同测试，另有 6 项碰撞 CPU 检查通过；这些检查结果可作为报告依据，按用户要求不提交测试源码。本次再次复核四个模型 SHA、选定课程构建与精简模块的 import；它们不代表完整仿真回放、硬件运行或全仓库测试通过。原训练器三处历史行尾空格保留原始字节，未为整理提交而改写在用源码。该批保存现有代码，不修改训练中的运行或实物仓库。
