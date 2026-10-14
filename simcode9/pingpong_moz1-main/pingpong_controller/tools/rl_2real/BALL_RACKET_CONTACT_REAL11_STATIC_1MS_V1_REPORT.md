# real11 静态首弹 1 ms 碰撞特性提交报告

独立特性版本为 `ball_racket_contact_real11_static_1ms_v1`，登记在 `TRAINING_CURRICULA.md` 的 Fixed-Plane V98 节。FPLC V98／V99 首次实现该版本；本次选定 V152、V151、V161、V172 的 checkpoint 配置均启用 `fixed_plane_v98_coarse_contact`，并保存相同 solref／solimp。它是碰撞合同版本，不能与旧 measured-QVEL V98、旧 GPU1 V153 或其他同号实验混淆。

## 精确参数及实现位置

| 字段 | 注册值 |
| --- | --- |
| XML | `tools/rl_sim/moz1_pd_1ms_v98.xml` |
| XML SHA-256 | `ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a` |
| 仿真球质量与半径 | `0.004 kg`，`0.020 m` 半径 |
| 物理步长与求解器 | `0.001 s`，implicitfast，100 iterations，elliptic cone |
| 拍面碰撞几何 | finite_mesh16 |
| pair condim／margin／gap | `3 / 0 / 0` |
| pair solref | `(0.006836, 0.228086)` |
| pair solimp | `(0.040000, 0.700000, 0.014298, 0.040000, 8.021856)` |
| pair solreffriction | `(0, 0)` |
| pair friction | `(0.13268182450948787, 0.11820649851379933, 0.0001544955787453423, 2.013852101985789e-05, 2.013852101985789e-05)` |
| 编译后拍面碰撞 geom 修正 | 沿法向 `+0.003 m`，保留 body 惯量与参考 site |

`mjx_juggle_env.py` 的 `V98_STATIC_CONTACT_PAIR_SOLREF`、`V98_STATIC_CONTACT_PAIR_SOLIMP`、`V98_RACKET_CONTACT_GEOM_OFFSET_M` 以及 V98 构造／编译几何路径共同应用参数；`train_juggle_mjx_curriculum.py` 的 V98 builder 与后继课程决定启用条件。**只复制 XML 不会复现接触**：临时 XML 归一化会重设源 geom 位置，3 mm 修正在环境编译后应用。XML 中可见的旧 geom solref 也不能替代显式 pair 的最终设置。

相关合同测试为 `test_fixed_plane_limit_cycle_v97_calibrated.py`、`test_fixed_plane_limit_cycle_v98_contact.py`；本轮 CPU 检查 6 项全部通过，包括编译后 pair、3 mm geom 修正及惯量保持。V97 的 62.5 µs 与拟合 solref `(0.005,0.165)` 是旧数值参考；V99 只改变历史采样，不另创碰撞参数。任何修改本表物理或数值参数均需新的碰撞特性版本。

## 静态证据及动态限制

该特性使用 11 次实物静态释放的首弹位置与时刻条件。登记的 GPU0 MJX 匹配释放检查给出首弹高度／接触时刻／apex 时刻 RMSE `5.38 mm / 1.04 ms / 6.72 ms`，最大误差 `10.02 mm / 2.0 ms / 14.6 ms`；41 相位高度标准差均值／最大值 `1.67 / 1.81 mm`。这是静态首弹结果，11 个案例均用于参数探索，不能作为全新独立动态留出集。

早期 V98 动态诊断必须保留：同一冻结 V96 actor 和配对种子，V96 物理 16/16 命中，V98 物理 0/16；单独移除几何修正仍为 0/4；只恢复旧 pair 或只恢复旧 cone 仍为 0/4，两者同时恢复才为 4/4。后继 FPLC 获得颠球证据并不消除这个早期负结果，也不能据此宣称真实动态拍面碰撞已完成标定。

本次只提交实际训练／validate 使用的 XML、环境实现和本报告，不提交旧标定搜索脚本、`calibration_source/`、中间候选、历史运行包或测试源码。较早静态位移拟合 V2／V3 的 `(0.012,0.10)` 与 `(0.0109,0.106)` 候选不是本表参数，不能混用；必要的静态结果与动态负面结论保留在本报告，不额外打包历史材料。

## 对实物代码的指导

实物推理不使用 MuJoCo pair 参数，不把有效数值 solref 当作机器人控制增益或材料恢复系数。3 mm 是仿真碰撞 geom 修正，不能据此给真实 FK site、球中心或相机深度统一加 3 mm。实物接触预测和 156-D actor 的几何上下文应对齐环境中 site／surface 的定义，并分别记录物理拍面、FK site 与球中心。

`policy_release_manifest.py` 的新 FPLC release 记录可保存该碰撞标签作为训练来源，硬件仍只接收 q_ref。未来真实碰撞更新使用有效入射／出射 3-D 球速、物理接触边沿的拍面 pose／twist、偏心、apex／XY 和独立留出数据；raw depth 与 surface-to-center depth 单独登记。当前特性归档保持原参数，不扩大接触 DR，也不修改真实控制增益。
