# simcode9 FPLC 实物探针部署

`RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 默认使用 `fplc_gpu1_v172`。
网页 Policy profile 中可选择下列五个独立模型；新模型不使用旧 GPU0 的 V85/V91/V97 子选择器。

| release / profile | actor | critic | 来源 |
| --- | --- | --- | --- |
| `fplc_gpu0_v152` | 156 | 477 | accepted release GPU0 |
| `fplc_gpu1_v151` | 72 | 393 | accepted release GPU1 |
| `fplc_gpu0_v161` | 156 | 477 | clean stability 后继 |
| `fplc_gpu1_v172` | 72 | 393 | clean commit 后继 |
| `fplc_v220_s17_u839_20261013_r1` | 72 | 477 | GPU1 V220，S17 small-DR consolidation，update 839 |

V172 的原文件名包含 `v159`，模型身份按 checkpoint 内的训练 profile 和 SHA-256 校验。
没有复制或另行导出模型。路径、哈希、观测 schema、动作语义和配置分别记录在
[`policy_release_manifest.json`](../config/policy_release_manifest.json) 和
[`fplc_model_contracts.json`](../config/fplc_model_contracts.json)。模型或 XML 字节不符会阻止加载。

V220 使用归档目录中的 `policy.pkl` 和对应的 `moz1_pd_1ms_v98.xml`，保留检查点中的
S17 DR 配置。归档状态仍为 experimental simulation checkpoint；本次接入验证模型
身份、接口及现有 FPLC 部署行为，不修改归档的训练毕业或实机认证状态。

## 配置检查与启动

只读检查在 ROS、相机和网页服务初始化之前退出；`PROBE_CONFIG_ONLY` 同时接受 `true` 和 `1`。

```bash
PROBE_CONFIG_ONLY=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
POLICY_RELEASE_ID=fplc_gpu0_v161 PROBE_CONFIG_ONLY=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
POLICY_RELEASE_ID=fplc_v220_s17_u839_20261013_r1 PROBE_CONFIG_ONLY=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

选择模型后沿用原部署流程启动。`ROS_PUBLISH_COMMANDS=true` 仅授权网页发送功能；
启动后的 Send command 默认关闭，复位完成或模型切换后需要操作者重新使能。

```bash
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
POLICY_RELEASE_ID=fplc_gpu0_v161 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
POLICY_RELEASE_ID=fplc_v220_s17_u839_20261013_r1 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

新模型要求 `fplc_direct` 预设、200 Hz 控制、原始一致的关节 q/dq。
球状态刷新默认 90 Hz，可在启动时选择 120/200 Hz；训练采样频率和模型字节保持原样。
不符合 release 的维度、频率和 dq 模式会拒绝启动。

```bash
# 兼容默认路线
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh --ball-state-rate-hz 90
# 相机使用 v1.4 当前时刻预测，向策略球历史提供 120/200 Hz 状态
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh --ball-state-rate-hz 120
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh --ball-state-rate-hz 200
# 环境变量等效；只读检查不打开相机、不发布命令
BALL_OBSERVATION_RATE_HZ=120 PROBE_CONFIG_ONLY=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

此选项只适配 FPLC 的球历史采样。相机采集及传感器状态发布仍为 90 Hz，
控制/关节反馈接收链路不改动。120 Hz 在现有 200 Hz 控制节拍上做分数采样，
刷新间隔为 5/10 ms，长期平均为 120 Hz；200 Hz 每个控制周期刷新。
相机的高频状态来自同一个有界 v1.4 快照查询，动捕使用其已到达的原生观测。
没有增加图像处理、历史拟合、发布定时线程或等待传感器的逻辑。

预测状态的表示时间与真实测量时间分开：球历史的帧龄对应位置表示时刻，
真实深度年龄另行传递并检查 350 ms 上限。预测不会把 `ball_observed` 设为 true，
不能初始化真实观测身份，也不能延长原预测有效期。网页显示启动时的实际选择；
策略运行期间不允许偷偷修改刷新频率，改变频率需重新启动。

2026-10-12 的调度修复默认启用 `CAMERA_SUBPROCESS=true`：直接 RealSense 检测、
追踪和 JPEG 可视化位于无 ROS 节点、无机器人发布器的独立进程。控制线程只读取
完整缓存帧，保留源时间戳、有效性和递增帧龄，不等待相机处理。`ros_topic` 相机
模式仍沿用原回调路径。FPLC 与 GPU0-QVEL/REAL-RMP 都使用独立 5 ms 控制线程；
控制子进程的 Python GIL 时间片为 1 ms，小型 CPU 网络默认使用单线程 BLAS。

这两类策略的网页诊断快照默认 30 Hz，网页请求/显示仍为 60 Hz，网络、训练投影、
完整控制记录仍为 200 Hz。镜像每个 sourceSeq 的结果完整保留，额外的网页镜像
状态副本最多发送 60 Hz。`PROBE_BACKEND_SNAPSHOT_RATE_HZ` 可显式覆盖诊断频率。
`CAMERA_SUBPROCESS=false` 用于回退排查，需要重新检查实际控制周期。

发令中的策略每 80 个周期检查真实墙钟；平均周期超过 5.4 ms，或单个周期超过
25 ms，会停止策略发送、执行已有的实测关节保持、作废 FPLC 历史并锁存重新授权
要求。该检查保留原 5 ms 积分，不通过补发突发指令或增加积分步长掩盖欠频。
它是欠频保护，不能使普通 Linux 调度成为硬实时系统。真实相机/ROS/录制/镜像
同时开启的不发令验证中，V172 网络及 q_ref 更新约 198.94 Hz；离线完整发送路径
连续 20 秒约 200.01 Hz。具体频率口径和限制见
[`相机隔离与 200 Hz 验证报告`](../analysis_results/simcode9_200hz_camera_fix_20261012/report.md)。

2026-10-13 的完整负载复测进一步发现动捕处理仍与控制共享进程，以及全量循环
垃圾回收的长暂停。现在默认 `MOCAP_SUBPROCESS=true`，动捕解析、标定、速度和
时钟同步在独立传感器进程中执行，控制读取带原始时间戳的完整缓存；过期、退出
和时钟异常仍失效。FPLC 必须预编译 Numba 投影后才能启动，CPU 数学库固定
单线程，独立控制子进程只在停发后执行循环垃圾回收。网页另显示活动策略频率，
避免把复位/停发后的空载频率当作实际策略更新率。

实时调度修复前，V172 的完整计算/独立 ROS 发布实测约 199.7–200.0 Hz；GPU0 V85
参考测试出现短窗口欠频。该轮全部测试不向机器人发令。另发现默认预测桥的一次恢复拒绝
导致相机控制输入失效，不能将其误判为检测丢失或欠频，亦不能据此宣布所有
部署问题已解决。数据、边界保护及验证范围见
[`完整发布链路与两路球输入复测`](../analysis_results/simcode9_publish_200hz_fix_20261013/report.md)。

新模型固定使用 `simcode9_native` 复位。下载代码中的 `right_arm_reset_degrees`
只用于 700 步暖机、冻结参考姿态；实际 `racket_launch` 回合随后使用
`racket_launch_racket_follow_episode_target_xy` 和
`racket_launch_racket_follow_fixed_contact_surface_z` 的原生 IK，
将球拍移至目标 (0, -0.490, 1.025) m 附近。当前五个模型的实际 FK 复位点约为
(-0.000300, -0.489780, 1.025167) m，处于缩小后的实物框内，无需扩宽。
网页锁定该复位选项，旧水平预设请求不会覆盖新模型复位点；模型切换会更新复位目标。
CPU 镜像同样先按原关节角暖机，再安装原生 IK 复位关节角。
原生 IK 保留暖机参考姿态及训练时 roll/pitch 的 ±2° 投影限制。

## 动作路径与状态

实物路径为确定性 actor 均值 → 裁剪归一化 QVEL → 训练时笛卡尔投影 →
持续 q_ref 单次积分 → 已有实物 RMP。
投影使用 q_ref 的 FK/Jacobian、实测 q 的关节边界、每关节 10 秒 reference horizon、
24 次加权 Dykstra 扫描、0.5 mm / 0.1° 内边距，再检查候选的精确非线性 FK。
单机器人路径使用启动时预编译的 Numba 循环加速；不使用 fastmath，扫描次数和次序不变。
float32、float64 及实际 FK 的混合精度输入，均以原 NumPy 运算逐值核对。
Numba 不可用时不能进入 FPLC 实物探针部署；NumPy 版本保留用于离线数值核对，
不作为实物发送的静默回退。配置值不代表实际频率，应检查实际计算和发布时间。
训练投影无法接受的候选保持旧 q_ref，previous_action 使用投影后的归一化速度。

`safety_limiter.filter/reset/observe_published` 不参与探针的策略、保持、停机或复位路径。
新模型强制关闭实物软件 RMP、PD、延迟队列、逆补偿、plant matcher、dq adapter、
actor 后的动作盾和观测缩放。MuJoCo 只用于 FK/Jacobian 与固定的 0.7 秒参考姿态暖机；
完整 RMP/PD 和接触参数仅存在于独立 CPU 仿真镜像。

72D 和 156D 使用各自的观测布局。历史每 5 ms 移动一格。默认兼容路线持有直接球位置，
按 90 Hz 分数时钟刷新并保持真实帧龄；选择 120/200 Hz 时，相机改用有界预测状态
刷新球历史，真实测量的标记和年龄另行保留，动捕沿用其原生位置及年龄。
156D 的 contact context 单独复现 site/surface 的区别和 apex 阶段。
实物 hit count 沿用相机速度反转事件的上升沿，不能将其当作仿真的真实碰撞传感器。

相机／动捕短时漏检采用与 GPU0-QVEL/REAL-RMP 相同的 **350 ms 有效状态上限**。
先有可信有效状态时，漏检期间保留其位置和速度，按实际经过时间与 5 ms 控制步长
的较大值累计年龄；原始 `rawValid` 和 FPLC 的直接 `ball_observed` 仍为 false。
FPLC 自己维护所选 90/120/200 Hz 分数时钟及 200 Hz 历史，外层只处理有效性，不再重复采样。
网页“漏检保持 q_ref”开关默认开启，控制传感器漏检时是否保持参考。
开启时 Send 保持开启、持续发送最后 q_ref，暂停生成新的网络 QVEL；
q_ref 不改写为其他位姿，previous_action 记录实际执行的零 QVEL。
可信新观测进入内部所选频率的采样后才恢复网络推理。
关闭时用有效的持有／预测球状态继续调用网络，按原训练投影正常更新 q_ref。
运行中切换在后端接受网页请求后的下一控制周期生效，不清空球年龄，也不解除故障锁存。
有效预测期间的漏检同样服从参考保持开关；预测算法与到期条件保持原有规则。
这是明确的参考保持，记录中 `fplcProjection.ballDropoutHold=true`、
`actorExecuted=false` 可辨识；不将漏检标志伪装为已观测。
记录中的 `policyBallHandover.state=held_dropout` 可区分持有状态与新观测。
连续丢失超时仍撤销 Send、丢弃策略历史并要求明确重新使能；恢复识别不会自动续发。
没有可信初始状态、非法／过期观测、预测到期和网页明确的无效 drag 输入不使用此容忍。
复位、模型或观测来源切换会清空持有状态；预测／估计器的有效期也不会被额外延长。

启动默认值可以通过环境变量设置；网页也可随时修改：

```bash
# 关闭参考保持：漏检时仍继续网络推理与q_ref更新
ENABLE_BALL_DROPOUT_REFERENCE_HOLD=false ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

# 开启参考保持（默认）
ENABLE_BALL_DROPOUT_REFERENCE_HOLD=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

对应Python参数为 `--disable-ball-dropout-reference-hold`，网页请求字段为
`enableBallDropoutReferenceHold`。启动、运行状态与录制均记录当前选择。
两个模式都保留350 ms无有效预测时的状态持有上限、预测到期保护及全部发送边界。

选定 V172 是 clean/pre-DR 模型，checkpoint 中 dropout 与相机随机漏检概率均为零。
上述改变已验证软件有效性、历史和发送边界，不能将 350 ms 软件上限当作该模型已通过
对应时长的实物漏检性能验收。2026-10-12 的录制回放、回归及其他缺口见
[`漏检修复与代码对比报告`](../analysis_results/fplc_dropout_review_20261012_194401/report.md)。

实物发送前再次核查不超过 10 ms 的关节反馈、实测和候选位姿、最终发送值与 actor
q_ref 一致性。外部安全介入或推理故障会保持实测位置、撤销发送、丢弃策略历史并要求
显式重新使能；硬边界故障锁存到复位。同一 tick 的旧发送请求不能重新启动命令流。
复位使用独立的受监督慢轨迹，允许从新工作空间外恢复；没有把复位输出偷偷写成策略 q_ref。
simcode9 的球边界 done 同样撤销 Send，发布一次新鲜实测位置保持，丢弃历史，锁存到显式复位。
恢复球输入或旧 Send 请求均不能自动恢复策略。停发、硬停和重新使能时，FPLC 不读写 limiter 状态、
不对实测保持值作额外裁剪，停发保持也要求关节反馈年龄不超过 10 ms。

## 实物工作空间

新训练硬框为 X=[-0.348, 0.348]、Y=[-0.700, -0.280]、Z=[1.000, 1.150] m。
根据操作者明确指定，实物停止边界采用下表，再与现有
`config_done_monitoring.yaml` 或显式物理边界取交集。
实物边界独立于训练硬框，发送前的实物位姿检查不再将 Z 缩回训练范围。
当前默认生效边界为：

| 轴 | 实物边界 / m |
| --- | --- |
| X | [-0.150, 0.150] |
| Y | [-0.600, -0.350] |
| Z | [0.950, 1.200] |

根据操作者要求，实物发送检查中相对于冻结暖机姿态的 roll/pitch 上限均为 ±4°，yaw 自由。
训练投影仍为 ±2°，参考姿态不变；这两个上限在启动日志和网页分别显示。
训练投影仍使用 checkpoint 的原始硬框，不随实物范围扩大而改变。
实物框只执行显式拒绝/保持/停机，
不会通过 `safety_limiter` 或额外投影改写策略输出。
四个 simcode9 模型共用此实物上限；环境变量或网页请求只能进一步缩小。
网页中的新模型锁定硬边界模式，伪造 `off` 请求也不会关闭检查。
网页边界图和记录分析使用当前模型的维度与字段，不将历史 XYZ 当成速度。

## Send command 自动关闭的排查

网页 `command rearm` 及中文说明显示撤销原因和触发时的关节反馈年龄/允许上限，
终端同时输出一次 `[CommandSafety][STOP]`。这些诊断保留触发时的数据，
不会因下一条新反馈到达而掩盖原因。必须处理原因后显式重新使能。

2026-10-10 在机器人内置仿真模式实测 `/joint_states`：三秒收到 61 条消息，
相邻消息的接收间隔中位数约 0.050004 s，header 间隔约 0.050000 s，发布者为
`/MCMessageNode`。本次复现的撤销原因为 `reset_arm_fresh_joint_feedback_required`，
触发时反馈年龄 49.6 ms，而新模型要求不超过 10 ms。
`record_new3/4/5` 的六段历史记录中独立关节采样间隔约 5 ms（200 Hz），
说明当前发布行为与这些历史记录不同；仍需核对控制端实际运行文件/配置才能确认原因。
不能通过放宽实物新鲜度阈值、重复发布旧 JointState 或伪造时间戳来掩盖 20 Hz 反馈。
若 `/joint_states` 没有发布者，应先检查控制端服务是否在线；网页的 200 Hz 控制
定时器不代表底层关节反馈也按 200 Hz 发布。

同日晚再次开启内置仿真时，`/MCMessageNode` 的 `/joint_states` 恢复到约 200 Hz。
在线拖球对比、done/丢球/指令租约停发记录及剩余调度问题见
[`内置仿真调试报告`](../analysis_results/simcode9_builtin_drag_20261010_214606/report.md)。

2026-10-15 针对 `probe_recording_20261014_234231` 的约 12.1 ms 接收间隙，
将关节反馈移至独立 ROS 节点及单线程 executor；Cartesian/视觉回调继续使用原 executor。
接收线程通过 `PROBE_JOINT_RT_PRIORITY`（默认 1）单独配置 SCHED_RR 和性能核亲和性，
与策略执行线程的调度设置相互独立。JointState 回调缓存关节名索引，只解析右臂七关节。
接收与执行均以 200 Hz 为目标，策略及投影仍使用真实输入按原 5 ms 时钟执行。

2026-10-16 按操作要求将 FPLC **运行中的**推理与发令反馈年龄门限统一为默认 15 ms；
可用 `FPLC_JOINT_FEEDBACK_MAX_AGE_MS=10|15|20` 启动对照，合法范围为 10–20 ms。
启动使能、复位使能及停止位置保持仍要求 10 ms，GPU0/V153 门限保持 10 ms。
JointState 订阅深度由 10 改为 1，调度恢复后直接读取最新原子 q/dq，避免排空旧反馈。
年龄在配置门限内时直接执行本拍，不增加等待、不改写接收时间、不外推关节状态。
超过门限时，只在当前控制周期剩余预算内释放锁/GIL 等待新回调；推理前预留 3 ms，
发令前预留 0.5 ms，单次等待最多 5 ms，绝对等待界限为旧样本接收时刻加“门限 + 5 ms”。
等待超时、持续中断、缺失反馈、源时钟异常或位姿越界仍触发原有停止流程。
复位和启动使能所需的新鲜度检查不进入抖动等待。

快照 `ros.jointFeedbackReceiver` 提供接收间隔、超 10 ms 次数、等待恢复/超时次数及
接收线程调度结果；其中 `cadence` 给出最近 200 个真实回调间隔的频率、P99 和最大值。
`receiptIntervalsOver10ms` 始终统计原始 10 ms 长尾，`receiptIntervalsOverLimit` 统计配置门限；
`lastWait` 记录最近一次等待的起始年龄、实际可用预算、耗时、结束年龄及恢复结果。
故障帧即使没有推理 trace，也保存同一控制 tick 捕获的关节时间信息。
`backendControlLoop.cadence` 与 `backendControlLoop.policyPublications` 分别统计
控制周期及实际策略发令间隔，不能将平均 200 Hz 单独作为稳定性结论。
硬停止的 `holdAttempted` 表示已尝试保持，`holdSent` 只在保持指令实际发送成功时为 true。

本次 20 秒隔离 ROS 基准完成 4000 次真实 FPLC 推理及测试话题发令：接收
199.999 Hz、发令 200.003 Hz，间隔 P99 分别为 5.155 / 5.441 ms，无超时。
随后真实机器人只读测试中，15 秒完成 3000 次本地推理：接收 199.993 Hz、推理
199.998 Hz，间隔 P99 为 5.146 / 5.533 ms，最大反馈年龄 3.544 ms，无超时。
实机测试未发送指令；这些结果尚未覆盖相机、镜像、录制同时运行的完整负载。
原始指标及复现脚本保存在
[`joint_feedback_200hz_20261015`](../analysis_results/joint_feedback_200hz_20261015/)。

完整负载复测脚本与原始记录位于
[`joint_feedback_full_load_20261016`](../analysis_results/joint_feedback_full_load_20261016/)。
测试启动器仅把正式脚本的 Python 入口改为测试夹具，保持相机、动捕、镜像及录制配置；
发令始终限制在 `/probe_feedback_validation_20261016/commands`，不会发到机器人话题。
静止机械臂无法跟随测试目标，因此各轮使用纯软件复位，并保留实际安全停止及重入记录。
同时修复控制子进程启动配置在 maxsize-one 状态队列被快照覆盖的竞态：收到父进程
确认前，合并状态始终携带配置；确认后恢复普通快照，运行期不重复发送配置。

后续优化将 FPLC 位姿查询改为等价的运动学计算，仅训练投影计算雅可比，并使用独立
5 ms 控制线程。四个模型各 500 步的观测、动作、q_ref 及历史逐项完全一致；内置仿真
drag ball 两种输入各 20 秒实测发令约 199.6 / 199.4 Hz，镜像和录制同时开启。
安全阈值、done 锁存和停发机制保持生效。短时调度抖动仍存在，详细记录见
[`200 Hz 优化报告`](../analysis_results/simcode9_200hz_optimization_20261010/report.md)。

2026-10-12 重新检查实际停止运动：软件停发/锁存成功，但无效球停发后球拍继续移动
约 85 mm，实测 pitch 达 5.74°，超过 ±4°。当前的一次位置保持不能视作可靠运动停止；
恢复 RMP 的停止重放也尚未完全复现实际偏差。普通运行的 MuJoCo 差异主要表现为
约 21–22 ms 滞后。详细数据、时基校正和停止风险见
[`内置仿真与 MuJoCo 差异报告`](../analysis_results/simcode9_builtin_gap_20261012_141646/report.md)。

## 编译与验证

2026-10-13 更新：相机与动捕现在都在启动时对齐到控制端 ROS 时钟，网页换源后
重新采集至少 8 个不同时间戳、跨度至少 50 ms 的样本。换源清空预测、估计器及
球采样历史，正在发送时关闭发送并要求显式重新使能。页面显示 `Ball clock`
状态；策略启动前等待 `synchronized`。慢复位保留独立于球时钟的安全恢复流程。

直连 RealSense 使用 IR 帧自身的时间戳和编号。global/system 时间映射保留采集
延迟；global 时间估计小幅领先电脑时，只向过去校准，映射后的观测不能位于
未来。hardware 时间使用滚动最小单向偏移，其固定传输延迟不可辨识。ROS 图像
话题保留有效 header 时间。动捕继续使用原来的滚动偏移同步，传感器子进程的
接收时钟由实际 ROS 时钟锚定。原始时间戳、映射时间戳和 epoch 写入记录；
轮询、换 marker、重复帧、工作进程退出均不能让旧数据变新。

控制进程缩短 GIL 时间片，并在混合 CPU 上选择性能核。网页快照在控制周期
剩余预算不足时延后；网络、原 24 次投影、5 ms 积分、安全检查和控制录制仍
逐周期执行。实物工作区、姿态限制、10 ms 关节新鲜度及欠频停止阈值均未放宽，
FPLC 没有增加 safety_limiter 调用。

完整调用链使用 `perf_counter_ns` 与 `thread_time_ns` 成对计时，并采集 Linux
`/proc/thread-self/schedstat` 的就绪队列等待时间。下表来自修正前 3,223 个活动
周期，包含计时钩子成本。网络、投影包含在 `predict` 中，`predict` 又包含在
策略控制步骤中；这些嵌套行不能相加。快照只在部分周期执行，其均值按全部
活动周期计算。

| 阶段 | 平均实际耗时 ms | 平均线程 CPU ms |
| --- | ---: | ---: |
| 网络推理 | 0.044 | 0.043 |
| 笛卡尔投影与候选参考检查 | 0.499 | 0.390 |
| 完整 `predict`（观测、网络、投影、历史） | 0.856 | 0.648 |
| 策略控制步骤（含 `predict`） | 1.402 | 1.045 |
| 发布边界检查及 ROS 发布 | 0.523 | 0.378 |
| 其中真实 ROS 发布 | 0.219 | 0.171 |
| MuJoCo 镜像通信 | 0.133 | 0.098 |
| 控制录制 | 0.391 | 0.244 |
| 网页快照（按全部周期均摊） | 0.146 | 0.065 |
| 探针主锁等待 | 0.006 | 0.001 |
| 完整周期 | 2.944 | 2.065 |

最慢活动周期实际耗时 13.956 ms，CPU 计算 2.677 ms，内核记录的就绪队列
等待 8.290 ms；剩余约 2.99 ms 包含其他阻塞，不能直接全算作 GIL 或归咎于
某个外部程序。队列 C 序列化 CPU 最大约 0.315 ms。真正的问题是调度尾部
延迟超过 5 ms 周期，造成漏掉控制槽及 184.36 Hz 的短窗口欠频；平均计算
成本没有耗尽预算。此 FPLC actor 实际采用等价 NumPy CPU 前向，模型名称
中的 GPU0/GPU1 是训练来源；本轮不存在 GPU 推理耗尽 5 ms 的证据。

代码 review 发现旧性能核选择只取绝对最高睿频，在本机仅允许 CPU 8、10，
排除了六个 5.7 GHz 性能核。现按硬件频率组选择全部八个性能核，并避免重复
选择同一物理核的超线程。相同夹具对照的平均周期为 2.567 ms，CPU 2.013 ms，
就绪队列等待均值从 0.081 降到 0.025 ms，P99 从 2.953 降到 0.123 ms。
该对照段 11.89 秒约 199.75 Hz，随后因球时钟重新同步停发，未发生欠频停止。
完整逐调用原始时间、独占/包含成本及复算脚本见
[`stage_metrics.json`](../analysis_results/simcode9_native_200hz_alignment_20261013/stage_metrics.json)、
[`stage_costs.csv`](../analysis_results/simcode9_native_200hz_alignment_20261013/stage_costs.csv)；
图见 [`stage_costs.png`](../analysis_results/simcode9_native_200hz_alignment_20261013/stage_costs.png)。

在用户确认外部控制关闭、允许原生话题完整策略测试后，最终版本在真实
`/mx_mix_command` 上完成以下两个 60 秒测试。独立观察器收到的所有目标与
发布账本逐条数值及顺序完全一致；两段均无欠频停止、录制丢样或漏检参考保持。

| 模式 | 网络及投影重算 Hz | 实际发布 Hz | 独立接收 Hz | 完整收到的指令 |
| --- | ---: | ---: | ---: | ---: |
| 相机 | 199.942 | 199.941 | 199.942 | 11,997 |
| 动捕（纯计算容量） | 200.000 | 200.001 | 200.001 | 12,001 |

用户明确允许暂时关闭不影响频率测量的限制。相关放开仅在分析目录的测试
夹具中生效。相机段临时取消候选 q_ref 实物位置箱，并关闭 done/丢球/预测桥。
动捕容量段集中关闭候选位置箱、NoSend 预览候选越界锁存、done/丢球、捕获/
释放、预测桥、漏检保持 q_ref、350 ms 球龄和两处时钟就绪停发门。仍使用真实
编码器与原始传感器时间，保留原网络、24 次训练投影、实测工作区和姿态检查、
10 ms 关节新鲜度及欠频检测。每次发布另检查新鲜机器人状态，必须确认外控
关闭；没有调用开启外部控制或复位的 SDK 服务。关闭项写入每段的
`nativeValidation`，部署源文件中的默认保护未放宽。原始失败段也保留，未用于
冒充通过结果。

需要区分“每拍重新计算 q_ref”与“每拍 q_ref 数值不同”：相机段两者均约
199.942 Hz。动捕容量段真实网络与投影每拍均运行，但原训练投影 10,850 次
选择 hold，数值变化率约 19.18 Hz；其 `currentSafe` 与 `candidateSafe` 均为真，
回退原因是 `maxLinearViolation` 超过原训练容差。例如首次回退为 0.0004093。
关节反馈未随参考运动、参考误差累积是本轮的条件，不能用这段比例推断实际
随动表现。未删减投影次数、放松训练容差或通过额外 safety_limiter 改写结果。

两段最大关节反馈年龄分别约 9.42 / 7.77 ms。实际周期存在抖动：相机 P99
8.74 ms、最大 14.54 ms；动捕容量 P99 6.98 ms、最大 9.46 ms，最慢 80 间隔
窗口分别约 192.56 / 197.73 Hz。结果验证了持续 200 Hz 的完整计算与真实话题
吞吐，不能承诺普通 Linux 每拍严格 5 ms。外控关闭且 SDK 执行回显为零，
机械跟随及 SDK 内部执行节拍没有被本轮验证。

动捕容量测试中保留了真实输入异常：时钟 epoch 从 1 到 9，球状态最大年龄
约 1.299 s；按授权关闭对应门后网络仍持续 200 Hz。这是容量验收，不能证明
开启全部保护的动捕部署不会因输入异常停发。另一个保留 350 ms 上限的段在
约 18.9 秒因 `policy_ball_state_fault` 正常停发。早期相机预测桥
`insufficient_history`、候选越界、球龄与时钟停发，均与欠频停止分开记录。

本机检查发现动捕路径经过 Wi-Fi，实际 `Power Management:on`。已在不重连
网络的情况下关闭，并把当前连接 `spirit-ai-radius` 的 powersave 配置改为
disable，防止重连恢复；前后状态见
[`wifi_power_save_fix.json`](../analysis_results/simcode9_native_200hz_alignment_20261013/wifi_power_save_fix.json)。
这项调整未消除上述全部输入异常。成功解析球包的内核投递间隔不等于全部
原始 UDP 包间隔，不能仅据此断言发送端或某个本机程序有问题。

记录及复核脚本位于
[`原生话题与同步验证`](../analysis_results/simcode9_native_200hz_alignment_20261013/native_metrics.json)。
最终再次关闭所有夹具放开项，以正常保护、Send 关闭验证网页 API 四次换源。
同步约 0.136–0.164 秒完成，命令发布总数为零，详情见
[`final_switch_alignment.json`](../analysis_results/simcode9_native_200hz_alignment_20261013/final_switch_alignment.json)。
CPU 调度、设备启动及数据流状态会影响实际等待时间，以页面同步状态为准。
完整回归 396 项通过；受沙箱限制跳过的 UDP 测试另行通过。最后相关模块复验
159 项通过，安装模块与源文件哈希一致。

2026-10-13 补充验收：用户明确要求实际运行
`RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 时的完整策略控制链维持 200 Hz。
最新启动脚本默认设置 `PROBE_CONTROL_RT_PRIORITY=1`，仅对独立策略进程中的
控制线程启用最低实时轮转优先级 `SCHED_RR`，并使用 8 个不同物理性能核心。
`SCHED_RESET_ON_FORK` 保证后续辅助线程及子进程不继承实时策略；没有修改
MovaX 控制器、原生 SDK 或机器人配置。本机实时优先级上限为 20，普通用户
可以启用该优先级。启动日志和网页 `backendControlLoop.threadScheduling`
记录实际调度结果。默认实时设置失败时关闭 Send，并报告
`control_realtime_unavailable`，不静默使用普通调度发送。优先级 0 仅用于显式
关闭实时调度的对照测试。

直接通过原始 `./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 启动并完成正常
Send + Reset + 复位完成后重新授权的网页 API 流程。相机、动捕、可视化、录制
及仿真镜像保持启动；使用 drag 球轨迹与独立进程提供的 200 Hz 模拟关节反馈，
指令只进入 `/offline_policy_rate_validation/launcher_command`。模型、投影、
实物位置箱、4° 姿态限位、350 ms 球龄、10 ms 反馈新鲜度、done 与欠频保护
全部保留；捕获/释放入口通过网页选项关闭，便于直接进入策略。没有绕过
候选 q_ref 越界保护或进入 `safety_limiter`。连续策略段 60.004 秒：

| 指标 | 实际启动脚本结果 |
| --- | ---: |
| 网络、映射及安全检查的策略计算 | 199.997 Hz |
| ROS 指令发布（消息原始发布时间戳） | 200.000 Hz |
| 独立 ROS 接收 | 199.999 Hz |
| 最慢 200 间隔策略窗口，约 1 秒 | 199.271 Hz |
| 最慢 200 间隔发布窗口，约 1 秒 | 199.357 Hz |
| 策略间隔 P99 / 最大 | 6.149 / 7.772 ms |
| 发布间隔 P99 / 最大 | 6.009 / 7.914 ms |
| 最大关节反馈接收年龄 | 7.559 ms |
| 异常停发 / 录制丢样 | 0 / 0 |

独立反馈进程隔离了测试客户端解码大网页快照对模拟编码器的影响。早期夹具
错误及反馈抖动导致的短段保留在分析目录，不用于替代上述连续验收。
另一个保留正常保护、每帧约 13.8 ms 相机计算负载的独立控制链测试连续 60 秒，
实际策略调用 200.000 Hz、发布 199.998 Hz，最慢一秒窗口 198.684 Hz。
这些结果验证实际启动路径在本机维持 200 Hz 的完整计算和发布，不代表普通
Linux 每个周期严格等于 5 ms，也不代表已经完成真实关节随动的长时验收。
实时相机在本次试验没有有效球检测；实时动捕短段仍出现时钟重新同步停发。
二者不是欠频故障，不能把 drag 连续验收称为两种传感器全部保护下的连续通过。

复算结果及脚本见
[`actual_launcher_metrics.json`](../analysis_results/simcode9_real_deployment_200hz_20261013/actual_launcher_metrics.json)、
[`verify_actual_launcher.py`](../analysis_results/simcode9_real_deployment_200hz_20261013/verify_actual_launcher.py)；
完整保护负载测试见
[`rt_full_60.json`](../analysis_results/simcode9_real_deployment_200hz_20261013/rt_full_60.json)。
已安装的 Python 模块包括此改动。重新启动原始脚本即可使用默认实时设置；
不要把隔离测试的模拟关节反馈话题配置带入实物部署。
本轮回归 402 项通过，因沙箱套接字权限跳过的 UDP 接收测试另行通过；Bash、
Python 和网页 JavaScript 语法检查通过。最终测试进程和原生命令话题发布者
均为空，详情见同目录 `final_tests.log`、`udp_test.log` 和 `final_audit.json`。

同日进一步直接启动原始脚本，选择 GPU0-QVEL/REAL-RMP V85，完成 60.010 秒
连续完整策略控制链测试。最新实时调度下，策略计算 200.004 Hz，指令发布
200.004 Hz，独立 ROS 接收 200.004 Hz，数值改变的 q_ref 同样为 200.004 Hz。
最慢一秒策略/发布窗口为 199.086 / 199.161 Hz；按控制记录复算的最坏 80 间隔
平均值为 5.051 ms，低于 5.4 ms。策略间隔 P99 / 最大为 6.363 / 9.034 ms，
无欠频停发、其他异常停发或录制丢样。该结果更新了修复前约 196.46 Hz 的
V85 比较结论，不把旧失败段作为最新性能。

GPU0 比较使用 drag 球、独立模拟编码器和隔离命令话题，同时保持实际相机/
动捕后台处理、可视化、录制与仿真镜像开启。保留位置、球龄、反馈新鲜度、
done、欠频保护及正常复位后重新授权；捕获/释放入口通过网页选项关闭。
只在隔离测试的环境中使用 GPU0 历史位置箱 X[-0.348,0.263]、Y[-0.702,-0.230]、
Z[0.842,1.462] m，启动后选用其原有水平复位。没有改动实物/FPLC 工作区配置，
没有发送原生实物命令。结论限于 V85 的完整软件控制链，V91/V97 未完成本轮
重新验收，真实随动亦未验收。原始数据、复算脚本及最终进程审计见
[`GPU0 V85 实际启动脚本验证`](../analysis_results/gpu0_v85_actual_launcher_200hz_20261013/actual_launcher_metrics.json)。

[`simcode9/COLCON_IGNORE`](../simcode9/COLCON_IGNORE) 排除整个仿真快照。
`colcon list --base-paths .` 只发现实物工程的四个包，`pingpong_controller` 已成功构建。

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select pingpong_controller
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src/pingpong_moz1 \
  python3 -m pytest src/pingpong_moz1/test/test_simcode9_fplc_contract.py -q
```

新增离线测试对照下载代码的原始观测、目标几何、投影数值函数和 CPU XML 暖机流程，
并覆盖发送边界、重新使能、网页切换、记录字段及镜像输入。当前机器没有 JAX，
数值配对通过 NumPy 执行原函数定义，不代表已经完成 MJX/JIT 全环境对拍或实物动作验收。
旧回归有两项在改动前同样失败：V85 停车测试缺少反馈 getter 的 fixture，
以及 V153 观测一项浮点差异为 1.1920929e-7、超过旧测试的 1e-7 阈值。

指导来源为 simcode9 的
[`FPLC_REAL_ROBOT_MIGRATION_REPORT.md`](../simcode9/pingpong_moz1-main/pingpong_controller/tools/rl_2real/FPLC_REAL_ROBOT_MIGRATION_REPORT.md)
及同目录的 Cartesian、simulation/policy 和 selected-model 合同报告。
