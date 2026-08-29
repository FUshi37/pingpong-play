# record_new3/4/5 球—球拍碰撞参数辨识

## 结论

从 65 段记录中最终保留 **116 次**连续、低残差且满足球拍几何的实物碰撞。
实物法向恢复系数中位数为 **0.665** （P10–P90: 0.556–1.031）。
对 simcode7 的独立球/圆柱碰撞做拟合后，推荐 MJX 球拍接触 pair 使用 `friction=0.3500`, `solref="0.003000 0.1525"`。

球质量按用户提供的实物质量直接采用 `0.0040 kg`。record_new3/4/5 本身仍不能独立反推质量：自由飞行重力加速度与质量无关，给定刚性球拍轨迹时碰后速度映射也基本质量不变，且记录中没有接触力/冲量；因此该值的证据来源标为实物先验，而不是视觉轨迹辨识。

## 数据筛选

- 每个 `cameraStampS` 只保留一次，排除 200 Hz 控制 tick 的重复观测。
- 用 `capture_time = sourceTimeS + cameraStampS - wallTimeS` 消除相机结果到控制循环的延迟。
- 碰撞前后各 6 个新相机点分别按固定重力弹道做鲁棒拟合，不让速度反转跨入同一窗口。
- 排除深度拒绝、深度分歧大于 70 mm、相机间隔不连续、拟合 RMSE 大于 6 mm、轨迹在碰撞点不连续、以及球心/球拍平面几何不成立的候选。
- 球拍位置、法向、线速度与角速度由原始 `ros.jointPosition/jointVelocity` 对应的 RMP trace 做 MuJoCo FK/Jacobian 计算；没有使用策略观测中的 dq adapter。

| 数据组 | 入选碰撞 | 恢复系数中位数 | 有效冲量摩擦比中位数 |
|---|---:|---:|---:|
| record_new3 | 71 | 0.662 | 0.043 |
| record_new4 | 37 | 0.658 | 0.059 |
| record_new5 | 8 | 0.985 | 0.042 |

这里的“有效冲量摩擦比”是球线速度跳变量的切向/法向比，不等同于胶皮的库仑摩擦系数：记录没有球自旋，因此切向冲量会与未知自旋耦合。`friction=0.35` 保留 simcode7 先验，它给出的 MJX 有效冲量摩擦比中位数与实物接近；不宣称这是已独立识别的胶皮材料系数。

record_new5 只有 8 次通过严格筛选，其恢复系数中位数 0.985 明显高于 record_new3/4 的约 0.66。因此它只用作留出检查，不用少量样本拉动标定值；该差异可能来自球体/实验工况变化，也可能是残余观测偏差。

## 仿真验证

拟合使用 record_new3/4，record_new5 完全作为留出验证。独立实验复用了 simcode7 的球半径、球拍圆柱、`margin`、`solimp`、`condim` 和 0.2 ms 内部步长。

| CPU MuJoCo 代理指标 | simcode7 原值（全部） | MJX 定标值（训练） | MJX 定标值（record_new5 留出） |
|---|---:|---:|---:|
| 碰后 3D 速度 RMSE (m/s) | 0.9429 | 0.2370 | 0.3094 |
| 法向速度 RMSE (m/s) | 0.9319 | 0.1806 | 0.2752 |
| 切向速度 RMSE (m/s) | 0.1437 | 0.1534 | 0.1415 |

## 可辨识性边界

- 球质量灵敏度实验测试 2.7 g, 3.6 g, 4.0 g, 4.4 g；相对 4.0 g 的最大碰后速度差见 `summary.json`。这说明轨迹数据不能给出独立质量估计，但仿真已按用户提供的 4.0 g 实物先验设置。
- CPU 代理模型的 5% 近最优 `solref` 时间常数范围为 3.57–10.00 ms，阻尼比粗网格范围为 0.20–0.20。90 Hz 相机不能分辨真实接触持续时间，因此最终保留 3 ms 时间先验，仅在 MJX 中直接拟合恢复敏感的阻尼比。
- CPU 代理模型的 2% 近最优摩擦范围为 0.57–0.80，而 MJX 的逐样本切向速度目标会把摩擦推向低边界；这种后端依赖性证明无球自旋观测时摩擦不可辨识。因此推荐值保留 simcode7 原先的 0.35，不使用边界解。
- `solimp=0.90 0.995 0.001`、`margin=0.001`、`condim=3` 保持不变；现有数据不足以同时辨识这些量。
- 没有球角速度时，无法分别识别胶皮摩擦和入射自旋。若要继续提升切向一致性，需要双相机/高速相机上的球面标记来估计自旋。

## MJX 交叉验证

对入选的 116 次碰撞在相同初值下分别运行 CPU MuJoCo 与 MJX；两者碰后速度 RMS 差为 **0.2465555 m/s**，最大分量差为 **1.2739280 m/s**。
MJX 对实物碰后 3D 速度 RMSE 为 **0.4642 m/s** （CPU MuJoCo：0.2427 m/s）。
MJX 训练集法向 RMSE 为 **0.1763 m/s**，预测恢复系数中位数为 **0.649**。
MJX 预测的有效冲量摩擦比中位数为 **0.060** （实物 record_new3/4 约 0.051）。
CPU MuJoCo 与 MJX 在高速、偏心、有摩擦的圆柱接触上不是逐步等价；因此最终阻尼比来自 MJX 直接回归，而不是 CPU 值。切向 RMSE 仍受未观测自旋支配。

## simcode7 落地

- 确定性接触：球 `mass=0.0040`, `size=0.02`；球拍胶皮 `size=0.075 0.001`；pair `friction=0.35 0.002 0.0001`, `solref=0.003 0.152540`, `solimp=0.90 0.995 0.001`, `margin=0.001`, `condim=3`。
- `JuggleConfig` 与非 curriculum 训练/验证 CLI 的 `dr_ball_mass_range` 已改为 `(0.0036, 0.0044)`，接触阻尼随机化默认值为 `(0.10, 0.24)`。
- 已同步 `models/moz1_pd.xml`, `tools/rl_sim/moz1_pd.xml`, `rl_juggle_env_random.py`, `mjx_juggle_env.py`, `mjx_smoke.py`, `train_juggle_rl_random.py` 和 `validate_juggle_rl_random.py`。
- 旧 checkpoint 及历史 curriculum profile 显式使用约 2.4–3.0 g 的球质量和 0.55–1.10 的阻尼范围，与新物理不同；为保持旧实验可复现，本次未批量改写历史 curriculum。使用这些 profile 时应建新的 sim2real contact profile，并重新微调。

## 产物

- `selected_impacts.csv`：每次入选碰撞的拟合、几何和相对速度。
- `recommended_contact_params.json`：可直接抄入 simcode7 的推荐参数与证据等级。
- `contact_fit_diagnostics.png`：实物分布、基线/拟合对比和 `solref` 目标函数剖面。
- `summary.json`：完整筛选计数、逐碰撞仿真结果和质量灵敏度。
- `mjx_validation.json`：CPU MuJoCo 与 MJX 逐碰撞交叉验证（运行验证脚本后生成）。

复现命令：

```bash
python3 src/pingpong_moz1/pingpong_controller/tools/rl_sim/identify_ball_racket_contact.py
PYTHONPATH=<mjx-deps>:src/pingpong_moz1/pingpong_controller/tools/rl_sim \
  python3 src/pingpong_moz1/pingpong_controller/tools/rl_sim/verify_ball_racket_contact_mjx.py
```
