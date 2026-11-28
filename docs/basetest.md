# MOZ1 实机底盘半天辨识测试清单与参数拟合脚本 — 实机端 AI 实现 Prompt

**日期**：2026 年 11 月 26 日  
**用途**：把下方 **第一部分（Prompt 正文）** 直接复制给部署在实物机器人仓库（`/home/yangzhe/Project/pingpong_playreal/pingpong-play`）上的 AI 编程助手（Codex / Claude Code / Antigravity），让其生成**只读接口探测工具、带多重硬安全保护的底盘辨识采集脚本、离线动力学与抖动拟合脚本、以及全套单元测试**。完成代码评审与单元测试后，操作员按 **第二部分（半天实机测试 Runbook）** 逐级监督执行。

---

## 第一部分：可直接复制给实机端 AI 的完整 Prompt

```text
你正在实物机器人仓库中工作：
/home/yangzhe/Project/pingpong_playreal/pingpong-play

目标：为 RSS 2027 移动颠球项目（Coupled Base–Impact Co-Planning for Agile Mobile Paddle Juggling）实现一套安全、可审计、可复算的 **MOZ1 实机底盘动力学辨识与拍面抖动耦合标定工具链**，供操作员在半天（约 3.5 小时）内完成无球真机采集，并离线拟合出可直接导入 MuJoCo/MJX 仿真及论文 Fig. 2 的实测参数与图表。

====================================================================
一、绝对安全红线与工程纪律（必须全部遵守）
====================================================================
1. 【严禁自动启动真机运动】本任务只允许读取仓库、编写代码/配置、运行只读接口检查（如可用）、运行离线回放与 pytest 单元测试。严禁在本次会话中向真实底盘、机械臂、腰部或头部发布任何非零运动指令，严禁启动训练或修改现有颠球策略/安全代码。
2. 【保留仓库现场】开始前先执行 `git status --short` 和 `git rev-parse HEAD`。保留所有用户已有改动；不得修改或放宽任何现有的 joint/EE/workspace/predictive-stop/feedback-stale/lease 安全逻辑。所有新增文件放入独立子目录（例如 `src/pingpong_moz1/pingpong_controller/tools/base_id/` 及对应 `src/pingpong_moz1/test/test_base_id_*.py`）。
3. 【严禁猜测 ROS 2 接口】不得凭空假设底盘控制 topic、消息类型（Twist / TwistStamped / 自定义 msg）、坐标系名称、里程计/IMU/动捕 topic 或手臂状态 topic。必须先检索当前仓库全部源码、launch 文件、URDF/YAML 配置（若本机 ROS 环境可用，可运行只读查询脚本列出接口），生成显式配置文件 `base_id_interface_contract.yaml`；凡是无法从仓库自动唯一确定的字段，必须标记为 `UNRESOLVED_REQUIRE_OPERATOR_FILL` 并让执行脚本在启动前 fail-closed 报错退出，同时支持 CLI/YAML 显式覆写。
4. 【Fail-Closed 与硬安全围栏】任何真机执行脚本必须默认 `--dry-run`（仅检查接口、校验参数、输出测试计划与理论位移边界，绝不发布控制指令）。只有同时满足以下所有条件才允许进入真机发布模式：
   - 显式传入 `--live`；
   - 环境变量 `CONFIRM_MOZ1_BASE_ID_LIVE=YES`；
   - 终端交互确认（输入当前测试阶段代号）；
   - 通过启动前自检（所有必需订阅源均在线、反馈新鲜、无 NaN/Inf、初始位置已锁存）。
5. 【实时运行保护（必须内建于采集节点）】
   - 软件地理围栏（Geofence）：以每次试次启动瞬间的底盘位姿 $(x_0, y_0, \psi_0)$ 为原点，一旦里程计或动捕位移超出配置边界（默认半径 `1.0 m`，可按场地调小不可盲目放大），立即中止试次并持续发送零速制动包直到速度归零。
   - 速度/加速度软钳位与斜坡保护：内置 `tier=low|med|high` 分级限幅表，严格校验输入指令不超过当前档位上限。
   - 反馈过期/心跳超时保护：底盘反馈或手臂关节反馈超过 `100 ms` 未更新、时间戳倒退或出现 non-finite，立即触发零速制动。
   - 优雅与异常退出保证：捕获 `SIGINT` / `SIGTERM` / 任何 Python 异常，在 `finally` 中以至少 50 Hz 连续发送零速命令 `0.5 s`（或直到实测速度 < `0.01 m/s` 且角速度 < `0.02 rad/s`），绝不残留非零速度指令。
6. 【禁止覆盖历史实验】每次运行自动创建唯一时间戳目录（如 `outputs/base_id/base_id_<YYYYMMDD_HHMMSS>_<stage>/`），若目标目录已存在直接报错退出。

====================================================================
二、需要辨识的核心物理与工程参数（6 大目标）
====================================================================
本工具链必须能够独立、定量地测量并拟合以下 6 项数据（全部采用 SI 单位）：

1. 轮型与运动学自由度审计（Kinematic & Cross-Coupling）：
   - 独立施加纯纵向 $v_x$、纯横向 $v_y$、纯偏航 $\omega_z$ 与斜向/旋转组合 $(v_x, v_y, \omega_z)$ 激励；
   - 量化主轴响应增益与非主轴串扰比（验证全向响应、舵轮转向死区/滞后，或非完整约束特征）。
2. 指令到运动的纯延迟 $\tau_{\text{base}}$ 与动态响应常数 $T_{\text{rise}}$（分 $x, y, \psi$ 三轴）：
   - 必须同时记录每条命令与反馈的 `wall_time_s`（单调时钟/墙钟）和 ROS `header_time_s`；
   - 显式检测并记录各传感源的 `header_time_s - wall_time_s` 常数时钟偏置（重要教训：本平台曾出现关节 header 钟与墙钟相差 ~6.094 s 的常数偏置，绝不可将其误算为物理延迟！）；
   - 使用互相关、起动阈值法（达到 10% 稳态响应时刻）及带纯延迟的一阶/二阶系统（FOPDT/SOPDT）分别拟合，并给出留出试次（held-out trial）预测 RMSE。
3. 速度、角速度与不对称加/制动加速度包络：
   - 分别辨识加速段最大加速度 $a_{\text{acc},x}, a_{\text{acc},y}, \alpha_{\text{acc}}$ 与急停制动段最大减速度 $a_{\text{brake},x}, a_{\text{brake},y}, \alpha_{\text{brake}}$；
   - 记录实际稳态速度上限 $(v_{\max,x}, v_{\max,y}, \omega_{\max})$、急停制动时间、制动滑行距离与速度过冲比例。
4. 底盘机动诱发的拍面位姿与法向抖动耦合（对移动颠球极其关键）：
   - 测试期间无球，右臂保持标准颠球待命位姿（静态 hold，不改变原安全逻辑）；
   - 同步记录：右臂 7 关节编码器 `q/dq`（经已有 FK 计算相对机体的拍面位置与法向变化）、底盘 IMU（机身俯仰/横滚角与角速度）、以及动捕刚体位姿（如有：底盘刚体与/或拍面刚体世界系位姿）；
   - 量化不同 $(|a_{\text{base}}|, |\alpha_{\text{base}}|)$ 档位下的拍面法向倾角抖动（RMS / P95 / Max，单位 deg）与拍面位置抖动（mm），并拟合抖动对加减速幅值的经验关系 $\sigma_\theta^{\text{base}}(|a|, |\alpha|)$。
5. 底盘控制与手臂 200 Hz 控制的并发同步性与时序抖动：
   - 在右臂正常维持 200 Hz 静态/微幅安全指令发布的同时，以标称频率发送低速底盘指令；
   - 统计手臂发布周期与底盘发布周期的 P50/P95/P99 抖动、丢帧/ZOH 保持比例、以及是否存在话题阻塞或控制器互斥。
6. 里程计 vs 外部动捕（如有）精度与延迟比对：
   - 同一试次内比对底盘内置 odom 与动捕计算的底盘线速度/角速度，输出标度误差、位置漂移率及相对延迟差，明确后续闭环控制器的首选底盘状态源。

====================================================================
三、必须交付的代码模块与文件清单
====================================================================
请在 `src/pingpong_moz1/pingpong_controller/tools/base_id/` 下实现以下 5 个模块（遵循 PEP 8、类型注解、4 空格缩进、SI 单位命名如 `_m_s`、`_rad_s`、`_ms`）：

1. `inspect_base_interface.py`（只读接口审计工具）：
   - 扫描仓库内与底盘/里程计/IMU/动捕/右臂状态/FK 相关的源码与配置文件；可选读取 `ros2 topic list -t` 快照（若无运行中的 ROS 节点则优雅降级为纯源码扫描）；
   - 输出 `base_id_interface_contract.yaml` 模板及人类可读的核对报告，明确标出已确认字段与 `UNRESOLVED_REQUIRE_OPERATOR_FILL` 字段。

2. `waveforms.py`（纯函数、无 ROS 依赖的辨识波形库）：
   - 定义分级安全测试序列（`tier=low|med|high`），每条波形返回解析时间戳序列、目标 $(v_x, v_y, \omega_z)$、以及理论最大位移/转角包络（用于启动前自动校验是否超出 geofence！）：
     - `step_vx_pos_neg` / `step_vy_pos_neg` / `step_wz_pos_neg`（正向加速→稳态保持→零速急停→静止稳定→反向回到起点，自动闭环防跑偏！）；
     - `ramp_accel_sweep`（多档斜坡加速度扫频，精测线性区与饱和拐点）；
     - `combined_diag_and_arc`（斜向平移 + 边走边转圆弧，检测轴间耦合与舵轮/全向约束）；
     - `prbs_or_multisine_small`（小幅伪随机/多正弦激励，作频域带宽验证与留出测试集）。
   - 提供预设安全参数表（默认值极度保守，可由配置文件覆写）：
     - `low`: $|v| \le 0.20\text{ m/s},\ |\omega| \le 0.30\text{ rad/s}$，单次理论行程 $\le 0.30\text{ m}$；
     - `med`: $|v| \le 0.50\text{ m/s},\ |\omega| \le 0.80\text{ rad/s}$，单次理论行程 $\le 0.60\text{ m}$；
     - `high`: $|v| \le 0.90\text{ m/s},\ |\omega| \le 1.40\text{ rad/s}$，单次理论行程 $\le 0.90\text{ m}$。

3. `run_base_id_experiment.py`（实机采集执行器，默认 `--dry-run`）：
   - 严格加载并校验 `base_id_interface_contract.yaml`；
   - 支持 `--stage`（`s0_idle_noise | s1_kinematic_low | s2_step_ramp_med | s3_step_brake_high | s4_combined_turn | s5_arm_concurrency`）与 `--dry-run` / `--live`；
   - 实时记录高频对齐数据流（推荐双轨存储：原始事件流 `raw_events.jsonl` + 结束时规整为压缩 `aligned_timeseries.npz`）与 `MANIFEST.json`（含 `git_head`、`git_dirty`、合同文件 SHA-256、CLI 参数、安全触发记录、每源消息计数与时钟偏置摘要）；
   - 复用仓库已有的右臂正运动学函数计算拍面位姿与法向（若找不到现有 FK 函数则明确报错并在文档指出，绝不手编一套未经校验的运动学链！）。

4. `fit_moz1_base_dynamics.py`（完全离线的参数拟合与论文出图脚本，不依赖 ROS 运行时）：
   - 输入一个或多个采集目录，执行严格数据审计：
     - 检验缺失帧、时间倒退、安全急停截断，并把全部接受/剔除试次及原因写入 `DATA_AUDIT.json`（不默默删坏数据）；
     - 自动分离时钟常数偏置，统一对齐到单调墙钟时间轴；
     - 拟合各轴 FOPDT + 饱和不对称加减速模型，区分训练集与留出验证集，报告 RMSE、P95、最大误差与 R²；
     - 计算急停制动距离与过冲统计；
     - 回归拍面法向倾角抖动与位置抖动对 $(|a|, |\alpha|)$ 的响应曲线；
     - 统计 200 Hz 手臂并发同步指标。
   - 输出三份核心产物（写入新输出目录，拒绝覆盖）：
     1. **`moz1_base_identified_params_v1.json`**（标准化仿真输入合同，必须包含字段：`schema_version`、`kinematic_type`、`axes.{vx,vy,wz}.{delay_s, time_constant_s, gain, v_max, a_acc_max, a_brake_max, overshoot_frac, heldout_rmse}`、`cross_coupling_matrix_3x3`、`racket_jitter_model.{normal_deg_rms_idle, normal_deg_per_mps2, normal_deg_per_radps2, normal_deg_p95_by_tier, pos_mm_p95_by_tier}`、`arm_base_concurrency.{arm_dt_p95_ms, base_dt_p95_ms, drop_rate, blocking_detected}`、`odom_vs_mocap`、`source_run_hashes`）；
     2. **`BASE_ID_REPORT_CN.md`**（简明中文技术报告，含全部汇总表、异常记录与物理结论）；
     3. **`fig_moz1_base_id_summary.pdf` 与 `.png`**（出版级双栏/四子图矢量图：A. $v_x/v_y/\omega_z$ 阶跃实测 vs 模型拟合；B. 加速与急停制动曲线及延迟标注；C. 底盘加减速 vs 拍面法向抖动散点与 P95 包络；D. 手臂/底盘并发周期分布）。

5. `src/pingpong_moz1/test/test_base_id_pipeline.py`（全面单元测试，纯离线可跑通）：
   - 合成测试：用已知真实参数（例如 $\tau_x=0.11\text{ s}, T_x=0.14\text{ s}, a_{\text{acc}}=1.2\text{ m/s}^2, a_{\text{brake}}=1.6\text{ m/s}^2$，叠加强制 `6.094 s` header 时钟偏置与高斯噪声）生成完整试次数据，验证 `fit_moz1_base_dynamics.py` 恢复延迟误差 `<5 ms`、加速度误差 `<5%`、不会把 6.094 s 偏置误判为物理延迟；
   - 波形测试：验证所有 `low/med/high` 波形零初速、零末速、且理论积分位移均严格位于 geofence 之内；
   - 安全测试：验证缺少 `--live` 或 `CONFIRM_MOZ1_BASE_ID_LIVE` 必报错；模拟越界、feedback stale、NaN、目录已存在时均 fail-closed 并触发零速序列；
   - 合同测试：验证 `UNRESOLVED_REQUIRE_OPERATOR_FILL` 未填时启动前必失败。

====================================================================
四、验收输出要求（本轮 AI 任务结束时向操作员汇报）
====================================================================
1. 列出本次新增的全部文件路径与 `git status --short`；
2. 给出 `inspect_base_interface.py` 的实际运行输出，明确指出哪些底盘/传感器接口已在仓库里确认，哪些需要操作员在现场补填；
3. 给出 `pytest` 全部测试结果（要求 100% 通过）与 `run_base_id_experiment.py --dry-run` 的示例输出；
4. 明确列出操作员在真机开跑前必须人工核对的 3 项现场检查项。
```

---

## 第二部分：MOZ1 实机底盘半天辨识现场测试清单（Runbook，约 3.5 小时）

代码在实物端通过全部单测与 `--dry-run` 后，操作员按以下 6 步顺序监督执行。**全程不放乒乓球**，右臂保持标准颠球待命位姿（静态 hold），场地保证至少 **`3.0 m × 3.0 m`** 平整无障碍，一名人员全程手握硬件急停（E-Stop）。

### 阶段 0：准备、接口核对与急停验证（30–40 min）
- [ ] **场地与电源核查**：确认地面摩擦均匀、无电缆缠绕（拔除或悬空网线/电源线）、确认电池电量 >40%、周围 2 m 内清场。
- [ ] **补全并锁定接口合同**：核对 `base_id_interface_contract.yaml` 中每一个 topic、消息类型、坐标系正方向（特别是底盘 `+X / +Y / +Yaw` 与右臂世界坐标系的定义是否一致！）。
- [ ] **静止噪声基线采集（`s0_idle_noise`，30 秒）**：零指令下录 30 秒，确认底盘 odom、IMU、动捕（若有）和右臂 `q/dq` 均在线、无丢帧，并记录静止状态下的拍面法向本底噪声与各传感时钟偏置。
- [ ] **硬/软件急停验收**：确认手握硬件 E-Stop 有效；用 `--dry-run` 确认每个档位理论位移都在 geofence 之内。

### 阶段 1：轮型、方向极性与低速单轴探底（`s1_kinematic_low`，30 min）
- **限幅**：$|v| \le 0.20\text{ m/s},\ |\omega| \le 0.30\text{ rad/s}$，单次位移 $\le 0.30\text{ m}$。
- [ ] 依次运行：`step_vx_pos_neg`（+X 后退回）、`step_vy_pos_neg`（+Y 后退回）、`step_wz_pos_neg`（+Yaw 后转回）。
- [ ] **现场核对 3 件事**：① 实际运动方向与指令正负号完全一致；② 横向 $v_y$ 是否可以直接平移（立刻确认全向/舵轮/差速轮型与舵轮转角死区）；③ 停止命令发出后是否干净刹停、无异常异响或剧烈点头。

### 阶段 2：中速阶跃、斜坡加速与急停制动辨识（`s2_step_ramp_med`，50 min）
- **限幅**：$|v| \le 0.50\text{ m/s},\ |\omega| \le 0.80\text{ rad/s}$，每项重复 **3 次**（用于留出验证与方差统计）。
- [ ] 运行三轴阶跃（`0.25 / 0.40 / 0.50 m/s` 及 `0.40 / 0.60 / 0.80 rad/s`），重点包含 **满速 → 零速急停**，采集加速段与制动段不对称性。
- [ ] 运行 `ramp_accel_sweep`（如 `0.3 / 0.6 / 1.0 m/s²` 与 `0.5 / 1.0 / 1.5 rad/s²`），找出速度跟踪开始滞后的加速度饱和拐点。
- [ ] 运行 `prbs_or_multisine_small`（低幅频域验证激励，作为离线拟合的独立留出集）。

### 阶段 3：高动态边界探顶与组合机动（`s3_step_brake_high` + `s4_combined_turn`，45 min）
> **进入本阶段前置门槛**：阶段 2 数据现场快速检查无异常晃动、底盘无打滑、拍面法向峰值抖动未逼近危险区。如出现明显轮子打滑或机身倾覆感，立即以当前档位作为安全包络上限，**绝不硬冲高速**。
- [ ] 按阶梯逐步上探高速阶跃与急停（例如 `0.65 → 0.80 → 0.90 m/s`；转向 `1.0 → 1.2 → 1.4 rad/s`），记录**安全不打滑、机身无剧烈摆动**的实际可用上限 $(v_{\max}, \omega_{\max}, a_{\text{acc}}, a_{\text{brake}}, \alpha_{\max})$。
- [ ] 运行 `combined_diag_and_arc`（斜向平移 + 边平移边 90° 转弯），检验 $v_x, v_y, \omega_z$ 并发时的轴间耦合、供电/电流限幅与拍面法向晃动最大值。

### 阶段 4：200 Hz 手臂并发同步压测（`s5_arm_concurrency`，25 min）
- [ ] 在右臂运行标准 200 Hz 安全静态 hold（及小幅安全正弦/微动测试）的同时，并发下发 `low` 与 `med` 档底盘指令（各 2 次，每次 15–20 秒）。
- [ ] 核对两路控制是否互相阻塞、手臂 200 Hz 周期 P95/P99 是否劣化、以及底盘启停瞬间手臂关节跟踪误差的峰值。

### 阶段 5：现场离线拟合与数据验收（30 min，拆场前必做！）
- [ ] 立刻运行 `python3 fit_moz1_base_dynamics.py <run_dirs...> --out <summary_dir>`。
- [ ] **拆场前 5 项硬验收**（只要有任意一项缺失，趁机器还在线当场补测 10 分钟）：
  1. `moz1_base_identified_params_v1.json` 完整生成且无 `NaN`/缺失字段；
  2. 留出验证集速度预测 RMSE 与延迟估计收敛合理（确认无 6 秒级时钟错位污染）；
  3. 明确拿到了 **轮型结论**、**三轴纯延迟 $\tau_{\text{base}}$**、**安全加速/制动上限**；
  4. 明确拿到了 **拍面法向抖动 vs 底盘加减速** 的实测曲线（这是路线 2 不确定性集 $\mathcal{U}_{k+1}$ 和 Stage 1 Fig. 2 的关键实测数据）；
  5. 确认底盘与手臂 200 Hz 控制可以稳定并发（或准确记录了通信/同步瓶颈，避免在 Stage 1 做出错误假设）。
- [ ] 打包备份 `moz1_base_identified_params_v1.json`、`BASE_ID_REPORT_CN.md`、`fig_moz1_base_id_summary.pdf/png` 及原始归档，准备带入新机器的路线 2 仿真环境。