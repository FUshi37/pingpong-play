# FPLC 训练运行与续训提交报告

此批保存四个选定课程的训练／validate 入口、对应启动脚本及其预检、完整状态恢复与主机资源保护。代码归档不会启动或停止任何训练，也不会授权复用另一个 W&B run。按用户要求不提交测试、旧独立实验模块、历史运行包或重复源码快照。

## 运行模块

`train_juggle_mjx_curriculum.py`、`train_juggle_mjx_ppo.py`、`validate_juggle_mjx_ppo.py`、`screen_juggle_mjx_checkpoints.py` 保存实际训练与模型验证入口。`run_with_host_memory_guard.sh`、`training_memory_scope.py`、`fixed_contact_oom_recovery.py` 保存资源保护；`freeze_training_source.py` 保存未来运行的源码／XML；`fixed_plane_completed_resume.py` 检查完整课程恢复。`fixed_plane_refinement_source.py`、`fixed_plane_ordered_updates.py`、V161／V172 preflight 检查选定来源、晋级证据和更新序列；V154 preflight 的数值源码归一化函数仍被后继预检调用，保留它及其依赖。

只保存 V151、V152、V161、V172 四个对应启动脚本。V153–V159 等课程模块是选定课程的父构建器或直接 import 依赖；V160 构建器被 V161 preflight 用作奖励一致性参考。保留这些文件不表示归档它们的模型、旧启动脚本或实验结果。

原物理 GPU UUID 分别为 GPU0 `GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`、GPU1 `GPU-e74458ec-002a-1bac-e0be-d0a5713b661e`。历史 launcher 的可执行配置与保存 checkpoint 为准，不能根据 run 文件名推断 actor、stage 或 DR。已有 formal jobs 保持只读。

继续训练应保留 actor、critic、Adam、真实 step、当前显式 stage 及匹配的课程记录；W&B-only offset 不改变 optimizer／checkpoint 的真实 step。对奖励、课程或 critic 的 versioned 改动应使用其专属迁移／锚点检查，不能一律重置或一律续旧 convergence window。新输出目录、GPU 所有权、主机内存和 78°C 温度检查仍由相应 launcher 管理。

GPU1 72-D 分支的 BC 输入 `outputs/rl_sim/v150_gpu1_preflight_20261006_r3/replay_obs_72d.npy`、`replay_actions_7d.npy` 随本批保存，合计约 12.86 MiB。它们是实际训练样本，不能当作额外模型；实物推理不读取它们。球质量声明和 RMP 输出 replay 报告一并保存。recovered-RMP JAX 实现已在 HEAD 中，本次只加入 canonical 配置；该目录也是 `freeze_training_source.py` 要求存在的源码根。不加入完整 ReactiveMotionPlanner 工作树。

## 复现范围与实物边界

当前共用代码和启动脚本描述所选 opt-in 训练功能。本次不提交重复 source_snapshot，不把当前源码标成每个旧模型当时的逐字节源码。四个 launcher 的原始启动仍需本地固定父模型、匹配 CSV、历史准入／配对 reward ledger；RMP replay 严格预检还核对原始 DataTracer、控制录制和 Movax 配置。用户本次只归档四个模型，不打包这些历史运行输入，因此本批不是所有旧训练任务都可从新 checkout 直接重跑的完整备份；不能删除或绕过原预检来掩盖缺失。

实物只需加载 actor 与注册配置，训练恢复用 critic、Adam、BC、主机 cgroup 和 W&B 不进入机器人控制链。实物 release manifest 绑定模型 SHA、观测 schema、QVEL／q_ref／投影参数及迁移报告；训练来源 UUID 和 true step 可作为追溯字段。

此前已检查启动脚本 `bash -n` 和 CPU 恢复合同。本次对精简后的模块集合检查 import、四个课程构建、Python AST、脚本语法和选定模型 SHA，不执行历史标定脚本、训练或 GPU probe。DR 中权重留待用户指定的后续提交。
