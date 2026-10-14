# GPU0 V97 实物试验模型增量说明

本批只替换 GPU0-QVEL/REAL-RMP 的 actor 模型，不改变已提交的实物运行时
边界。完整 57-D observation、200 Hz persistent bounded-`q_ref`、90 Hz
fractional 球状态和安全要求仍以 `GPU0_QVEL_REAL_RMP_GUIDE.md` 为准。

## 模型身份

| 项目 | 值 |
| --- | --- |
| profile | `goal_d455_measured_qvel_rmp_vertical_v97` |
| stage | `rmp97_rmp_internal_commit_12p5`（index 32） |
| stage/global update | `164 / 5329` |
| true step | `5836242944` |
| seed | `20261018` |
| actor / critic / action | `57 / 368 / 7` |
| checkpoint | `gpu0_v97_best_step5836242944.pkl` |
| SHA-256 | `f166a4b8adb6b5b6a846b14407baa5da32ba3535492d23cb0ad5978cece8568d` |

仓库路径：

```text
pingpong_controller/outputs/rl_sim/
selected_best_models_and_normal_reset_videos_20260901/gpu0_v97/
gpu0_v97_best_step5836242944.pkl
```

`gpu0_v97_model_release.py` 在反序列化前认证哈希，并校验模型身份、维度、
训练 profile、核心 V85 推理契约、固定底座和 4.0 g 球质量区间。

## 相对已提交 GPU0-QVEL/REAL-RMP 的变化

实物侧必须保持不变：

- actor 仍为 57-D 输入、7-D QVEL 输出，deterministic mean；
- 仍使用 measured `q/dq` 和 persistent bounded-`q_ref`；
- reference horizon 仍为 `[18,19,19,18,18,18,18] ms`；
- 发布的仍是 200 Hz `q_target`，送给机器人已有 RMP；
- 不在实物策略进程部署 recovered RMP、XML PD、仿真 output delay、执行器
  filter 或第二个 planner；
- observation 顺序、单位、球状态 represented timestamp 和 reset 原子状态均不变。

V97 新增的 `aligned_fixed` 仿真底座、球面落在球拍表面的 reset、4.0 g nominal
球和 `[3.9,4.1] g` 质量 DR 都是训练侧变化。它们不增加实物运行参数，也不允许
修改真实机器人已有的 RMP/低层控制器。

V97 的 stage/reward/curriculum 变化同样只影响训练和模型权重。实物合并时应以
新 profile/模型哈希替换 actor，同时复用既有 GPU0-QVEL/REAL-RMP adapter。

## 上机门槛

该模型是用户选择用于分阶段实物实验的 experimental checkpoint，不表示已经
通过实物验证。至少依次完成：哈希和维度校验、固定输入 golden-vector、逐字段
observation parity、离线 recorded replay、shadow target、无球低增益/限空间测试，
最后才进行监督有球测试。任一时间戳、encoder freshness、joint limit 或输出有限性
检查失败时必须 fail closed。
