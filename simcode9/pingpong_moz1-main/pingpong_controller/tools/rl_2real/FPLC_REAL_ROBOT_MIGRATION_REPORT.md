# FPLC 选定模型与实物代码迁移报告

2026 年 10 月 10 日。本次只归档用户接受的 FPLC GPU0 V152 R1、GPU1 V151 R1，以及沿相同 clean 血缘通过完整晋级的 GPU0 V161、GPU1 V172，各一个模型。四者均采用 7-D QVEL 动作。实物仓库现有 57-D／67-D adapter 不支持其 156-D／72-D 观测，需要独立版本的接口实现和回放测试。

## 模型身份

| 模型 | 已保存 stage 与 update | true step | actor／critic／action |
| --- | --- | ---: | --- |
| FPLC-GPU0-V152-R1 | fplc_v152_gpu0_s5_steady_quality / u2786 | 2112880640 | 156/477/7 |
| FPLC-GPU1-V151-R1 | fplc_v151_gpu1_s4_fourth_entry / u3187 | 883949568 | 72/393/7 |
| GPU0-V161-pre-DR | fplc_v161_gpu0_s7_clean_stability / u331 | 2297167872 | 156/477/7 |
| GPU1-V172-pre-DR | fplc_v159_gpu1_s10_clean_commit / u314 | 1334837248 | 72/393/7 |

- `pingpong_controller/outputs/rl_sim/releases/fplc_juggling_20261008_r1/gpu0_v152/policy.pkl`
  SHA-256 `5b98879f9a6f16e61ce2f32a3c5ac2164e9433afc2050fc4293e5c2e25277ffe`。
- `pingpong_controller/outputs/rl_sim/releases/fplc_juggling_20261008_r1/gpu1_v151/policy.pkl`
  SHA-256 `b34d33f66efe579bfc4c42d18226c1a05b3a273f0eec5b2722625b1c4d4b8c5f`。
- `pingpong_controller/outputs/rl_sim/fplc_v161_gpu0_mean_quality_formal_20261009_r1/07_fplc_v161_gpu0_s7_clean_stability.pkl`
  SHA-256 `27a2ac3c0ecf5cc5b0ff66353e8a1f46d8771164c2128d30820f5b986e37b559`。
- `pingpong_controller/outputs/rl_sim/fplc_v172_gpu1_quality_gates_formal_20261009_r1/10_fplc_v159_gpu1_s10_clean_commit.pkl`
  SHA-256 `96c9b8349fa700402a62d1c34e5f8ed2b968bab47cf2b9d3957a1db6cfccba4a`。

模型合同与 checkpoint 配置摘录在 `FPLC_SELECTED_MODEL_CONTRACTS_20261010.json`。四个文件约 37.17 MiB，均保留 actor、critic 和 Adam；不再另导出或复制一份权重。两个 accepted release 各有 1200-step、13-hit 视频，十次 steady apex 都在 1.40–1.45 m。clean 后继依据 rolling／advance／retention 三重晋级证据选择，未作所有模型的全新配对排名。

GPU0 clean 血缘为 V152 → V154 clean S7 → V161 clean S7；GPU1 为 V151 → V155 clean S8 → V156 clean S8 → V158 clean S8 → V159 clean S9 → V172 clean S10。V172 保留 `fplc_v159_` stage 名，因此文件名中的 V159 不等于模型 profile。GPU0 V161 的 DR 从 S8 开始，GPU1 V172 从 S11 开始；目录通用 best／last 和 DR 后再跑 clean 的权重均不在本批。

## 两种 actor 的精确索引

索引使用 Python 左闭右开切片。q、dq、q_ref_error、previous_action 保持训练的七关节顺序；位置 m，角度 rad，速度 rad/s。代码直接拼接这些值，不能套用旧 57-D／67-D 归一化。age 是源样本年龄秒数除以 0.25 后 clip 到 [0,1]。旋转 6D 是矩阵前两列依次展开为 column0 XYZ、column1 XYZ。

| 切片 | GPU1 V151／V172，72-D |
| --- | --- |
| `0:7` / `7:14` | measured q / dq |
| `14:21` / `21:28` | 持续 q_ref−measured q / 上 tick 投影后归一化动作 |
| `28:52` | 8 个 200 Hz 连续历史 tick 的持有球位置，最新到最旧 |
| `52:61` | 当前 FK racket site XYZ + rotation6D |
| `61:69` | 对应 8 个球样本 age / 0.25，clip |
| `69:71` / `71:72` | episode target XY / ball-position observed 标志 |

72-D 使用 `fixed_plane_v149.make_simple_position_observation`；最后标志在环境调用中是 `fixed_contact_ball_observed`，不能自行换成任意 predictor-valid 或 hit-valid 定义。

| 切片 | GPU0 V152／V161，156-D |
| --- | --- |
| `0:28` | 同样的 measured q、dq、q_ref_error、previous_action |
| `28:52` | 8 个最新到最旧连续 tick 的持有球位置 |
| `52:124` | 对应 8 个历史 FK racket site XYZ + rotation6D |
| `124:132` | 对应球样本 age / 0.25，clip |
| `132:133` / `133:135` | active desired absolute apex Z / episode target XY |
| `135:136` / `136:137` | ball_valid / clip(confirmed_hit_count / 4, 0, 1) |
| `137:140` / `140:141` / `141:142` | 预测触球球位置 / clip(time_to_contact,0,2) s / contact_valid |
| `142:145` / `145:151` | racket target XYZ / target rotation6D |
| `151:154` / `154:156` | target XYZ−当前 racket surface XYZ / next crossing XY |

156-D 是 `fixed_plane_v92.make_history_actor_observation` 的 137-D prefix，使用 V99 连续 lags `(0,1,2,3,4,5,6,7)`，再拼 `fixed_plane_v93.make_predicted_contact_context` 的 19-D。其 ball_valid 由 observed 与 state_valid 共同产生。contact context 来源是部署可用预测与 `fixed_contact_plane_training.py` 的目标几何；actor 没有直接球速槽不表示预测器不需要球速估计。真实 confirmed-hit 事件、apex stage 选择、预测已有传播标志、site／surface 差异均须逐项复现。

每 5 ms 推进历史，覆盖 0–35 ms；90 Hz 无新帧时持有位置，age 随时钟增长，不将一次真实新帧人为复制成多个新样本。reset／失效／重新接管时，按 frozen 环境初始化 q_ref、previous_action、球与拍面历史、age 和事件状态，必须有多 tick 配对证据，不能仅检查向量长度。

## 实物仓库修改位置

实物仓库为 `/home/yangzhe/Project/pingpong_playreal/pingpong-play`，本次只读核对。下列是后续具体修改建议，当前未实现或部署新的 FPLC adapter。

| 文件或模块 | 必须实现的变化 |
| --- | --- |
| `src/pingpong_moz1/pingpong_controller/policy_release_manifest.py` | 新 FPLC release 身份、四个 SHA、72／156 actor schema、QVEL 和独立配置校验；拒绝旧 profile 别名冒充新 actor |
| 同目录 `rl_policy.py` | 新 observation profile 和历史状态构造、reset、有效性、200 Hz／90 Hz 时钟合同；保留旧 adapter |
| 同目录 `mjx_rl_policy.py` | 加载对应 actor 结构与确定性 policy_mean，检查层尺寸；不要给输出再加未训练的 tanh 或探索采样 |
| 同目录 `policy_joint_sample.py` / `policy_ball_handover.py` | 原子 q／dq 与源时间戳、球观测持有、age／valid／预测模式；为 GPU0 提供可验证的接触事件和预测上下文 |
| 同目录 `pingpong_node.py` | 引入新 profile 的持续 q_ref、投影和回退状态；完成 sim／real 坐标、FK site／surface 配对；安全层后发布 `MechUnitCmd.jnt_pos` |
| `RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 和 `tools/interactive_policy_safety_probe.py` | 显式 FPLC 入口、release 校验、观测／指令影子回放；通过离线合同后才进行分级监督试验 |
| `test_policy_profile_contract.py`、`test_policy_release_manifest.py` 及新 FPLC 测试 | 错模型／错 schema 失败、各字段 golden vectors、多 tick reset／history／q_ref／投影配对、旧路径回归 |

四个保存配置均开启笛卡尔位姿投影；q_ref horizon 为每关节 10 秒，速度上限和 projection 常数见共用提交报告与 JSON。投影以持续 reference 的 FK／Jacobian 和 measured q 限制可行速度，随后更新 q_ref，并将投影后的归一化动作回填 previous_action。不能改为每 tick measured_q+action×lead，不能只复制积分公式而省略投影／非线性回退。安全介入时如何保留或重置 q_ref 属于新硬件合同，须明确并测试，不能默默把 simulator state 改成安全输出。

硬件链保持 policy q_ref → 既有安全层 → `MechUnitCmd.jnt_pos` → 机器人已有 RMP。不得再运行 recovered RMP、XML PD、仿真 output delay 或 actuator compensation。碰撞版本 `ball_racket_contact_real11_static_1ms_v1` 只绑定训练来源；3 mm geom 修正和 MuJoCo pair 参数不直接写到真实 FK、视觉偏移或机器人增益。

## 验收顺序与状态

先完成模型 hash／层尺寸／schema 校验，再做 frozen-input actor 和逐字段观测配对，再做连续 q_ref／projection／reset 影子回放。与真实时间戳、坐标、拍面定义、事件和预测完成配对后，按既有安全限制进行监督无球，再进入受限带球验证；本次没有执行这些硬件步骤。72-D 输入少，但仍缺 FPLC 专用实物适配与验证，不因维度较少就自动可部署。

当前 DR 尚在进行，只提交上述 clean 四模型。正式 RMP／PD encoder-feedback coverage、新球动态碰撞与真实观测噪声证据不能由静态 real11 标定或较高 return 替代。所选后继的完整晋级行见 `FPLC_CLEAN_STAGE_ACCEPTANCE_20261010.json`，指标摘要见 `FPLC_EVALUATION_AND_CONTACT_REPORT.md`。本次不提交重复冻结源码快照，训练时原始快照和未归档的中间父模型仍留在本地；当前共用源码不冒充每个旧模型当时的逐字节源码。
