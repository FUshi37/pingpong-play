# simcode10 模型实物探针接入

`RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 和网页的 Policy profile 支持以下七个独立模型。
启动默认仍为原来的 `fplc_gpu1_v172`；选择新模型时按 release 同时解析权重、XML、维度和配置哈希。

| POLICY_RELEASE_ID / 网页 profile | 训练版本 | Actor / Critic / Action |
| --- | --- | --- |
| `fplc_gpu1_v223` | V223 S15，姿态观测偏差 1° | 72 / 393 / 7 |
| `fplc_gpu0_v224` | V224 S14，V223 父模型分支 | 117 / 438 / 7 |
| `fplc_gpu1_v224` | V224 S14，V172 父模型分支 | 117 / 438 / 7 |
| `fplc_gpu1_v274_s9` | V274 S9，反弹外力 DR、球观测 DR 前 | 72 / 398 / 7 |
| `fplc_gpu1_v274_s11` | V274 S11，反弹外力 DR 加相机 AR1 | 72 / 398 / 7 |
| `fplc_gpu0_v319` | V319 S17，近侧位置与向内动量恢复 | 72 / 393 / 7 |
| `fplc_v321_s4_u2097_20261029_r1` | V321 S4/u2097，V172 长回合微调候选 | 72 / 393 / 7 |

例如：

```bash
POLICY_RELEASE_ID=fplc_gpu0_v224 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

只校验配置、不启动设备：

```bash
PROBE_CONFIG_ONLY=true POLICY_RELEASE_ID=fplc_gpu0_v224 \
  ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

V321 使用新归档的 `outputs/rl_sim/releases/fplc_v321_s4_u2097_20261029_r1/policy.pkl`，
SHA-256 为 `889eb976b18567d607ab9f86f35457c89f901e2ecf221b9391128cb471f0d001`。
启动和仅校验配置的命令：

```bash
POLICY_RELEASE_ID=fplc_v321_s4_u2097_20261029_r1 ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh

PROBE_CONFIG_ONLY=true POLICY_RELEASE_ID=fplc_v321_s4_u2097_20261029_r1 \
  ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

网页也可选择「simcode10 V321 S4 u2097 长回合候选 (72D)」。此版本与 V172 使用相同的
72D 观测、历史初始化、原生复位、200 Hz QVEL 积分和 24 次投影，独立校验新权重身份。
训练位置箱为 X[-0.348,0.348]、Y[-0.700,-0.280]、Z[1.00,1.15] m；
实物安全边界仍取现有 FPLC 范围与物理限制的交集。
归档的 1200 秒训练回合及奖励调整保存在 checkpoint 中，不改变探针的安全中断规则。
该权重为候选版本，归档记录的 S4 独立 1200 秒验证尚未完成；接入验证不等于实机效果验证。
接口依据见 [`V321 部署说明`](../simcode10/pingpong_moz1-main/pingpong_controller/tools/rl_2real/FPLC_V321_S4_U2097_DEPLOYMENT_20261029.md)
和归档目录中的 `release.json`。

新模型配置保存在 [`fplc_simcode10_model_contracts.json`](../config/fplc_simcode10_model_contracts.json)，
原 simcode9 合同文件保持原有内容及哈希。七个权重分别绑定各自 SHA-256，两个 V224 即使维度相同也不能互换。
XML 使用 simcode10 自带、与归档哈希一致的 `moz1_pd_1ms_v98.xml`。

72D 路径使用原 V149 位置观测，117D 路径使用独立的 V224 因果历史实现。
后者在 72D 前缀外缓存 41 个实际观测前缀，每个 5 ms 策略周期推进一次。
50/100/200 ms 三个 lag 各产生 15D：当前观测拍坐标系下的球位移、法向变化、
过去 `q_ref-q`、过去归一化球龄及有效性。历史不足或当前/过去 observed 无效时整块为零。
复位、安全干预和模型切换清空历史，并从新的测量状态重新建立。
部署使用标定 FK 和传感器输入；训练的姿态偏差、球噪声和外力 DR 不额外注入实物观测。

每个模型的训练投影范围来自其 checkpoint：V223、V321 沿用较大位置箱，其余五个使用
X[-0.145,0.145]、Y[-0.595,-0.355]、Z[1.00,1.15] m。
实物边界继续使用原 FPLC 实物安全范围与物理限制的交集，训练投影保持独立。
V319 训练时包含目标 Y 偏移，部署复位明确选择其中的零偏移 lane，目标保持 [0,-0.49,1.025] m。
网页复位继续使用兼容的 `simcode9_native` preset ID，显示为 simcode9/10 共用 FPLC 原生复位。

所有新 profile 复用现有 FPLC 的独立控制线程、关节反馈接收器、限时等待和调度配置。
控制、网络推理、24 次投影及历史推进保持 200 Hz；关节接收仍由独立 executor 处理。
球状态默认 90 Hz，已有的 120/200 Hz 启动选项继续有效；相机采集仍为 90 Hz。
网页后台快照默认 30 Hz，浏览器请求默认 60 Hz，控制记录仍按 200 Hz 生成。
模型在网页服务就绪前预加载；新增历史计算不引入定时器、同步设备调用或运行期模型加载。

网页可切换七个 simcode10 profile，显示对应的维度、观测段、投影范围和模型路径。
117D 的录制、离线对比和 CPU 仿真镜像使用对应观测索引，球位置历史不会被误读为球速。
模型切换继续执行停发、清历史、复位及显式重新使能流程。

[`simcode10/COLCON_IGNORE`](../simcode10/COLCON_IGNORE) 排除整个仿真快照，
`colcon list --base-paths . --names-only` 仅发现实物工程的四个包。
不需要构建 simcode10；启动脚本优先使用工作区源码。

离线回归命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src/pingpong_moz1 python3 -m pytest -q \
  src/pingpong_moz1/test/test_simcode10_fplc_contract.py \
  src/pingpong_moz1/test/test_simcode9_fplc_contract.py \
  src/pingpong_moz1/test/test_fplc_ball_dropout.py \
  src/pingpong_moz1/test/test_ball_state_multirate.py \
  src/pingpong_moz1/test/test_joint_feedback_receiver.py \
  src/pingpong_moz1/test/test_probe_timing.py
```

2026-10-29 V321 接入验证：模型合同、manifest、观测、仿真镜像、网页切换、
安全边界和丢帧回归 495 项通过，球状态多频率回归 53 项通过。
V321 的仅配置启动覆盖 90/120/200 Hz 球状态，拒绝错误的 117D 维度及 100 Hz 控制覆盖。
合成输入下运行 1000 个策略周期，网络与投影各执行 1000 次、球状态刷新 450 次，
平均控制频率 199.997 Hz；未启动设备或发送硬件命令。
本次计时与离线回归同时执行，结果见
[`simcode10_v321_runtime_20261029.json`](../config/policy_validation_reports/simcode10_v321_runtime_20261029.json)。

2026-10-27 离线验证：核心模型合同、镜像和丢帧回归 311 项通过。
球状态多频率、关节接收器、计时、manifest 和旧界面回归 249 项通过。
补充网页维度切换测试后，针对新模型、网页切换、旧界面和 manifest 的 209 项复验全部通过。
Bash/Python/网页 JavaScript 语法检查和 colcon 包发现检查通过。

实际执行网络、观测历史和 24 次投影的独立离线计时，每个模型运行 5 秒、1000 个策略周期。
各模型均调用网络和投影各 1000 次，球状态刷新 450 次，历史每周期推进。
本次各模型的 1000 次 q_ref 数值也均发生变化，没有用重复发送旧参考代替更新。
结果见 [`simcode10_runtime_20261027.json`](../config/policy_validation_reports/simcode10_runtime_20261027.json)：

| profile | 实际计算 Hz | predict P99 / 最大值 ms |
| --- | ---: | ---: |
| `fplc_gpu1_v172`，已有模型参照 | 199.996 | 0.631 / 0.838 |
| `fplc_gpu1_v223` | 199.997 | 0.841 / 1.136 |
| `fplc_gpu0_v224` | 199.998 | 1.041 / 1.392 |
| `fplc_gpu1_v224` | 199.998 | 0.724 / 0.835 |
| `fplc_gpu1_v274_s9` | 199.998 | 0.579 / 1.003 |
| `fplc_gpu1_v274_s11` | 199.998 | 0.545 / 0.854 |
| `fplc_gpu0_v319` | 199.996 | 0.543 / 0.683 |

复现命令：

```bash
python3 tools/benchmark_simcode10_policy.py --seconds 5 \
  --output /tmp/simcode10_runtime.json
```

该计时使用合成球和关节输入，未启动 ROS、相机或动捕，没有发送硬件命令，
不代表完整传感器/录制负载下的实机接收频率或关节跟随验收。
网络及 q_ref 重算频率与 q_ref 数值变化率分开记录；投影按原规则选择 hold 时仍然每周期计算。
控制线程、控制 tick、接收等待及反馈获取函数与接入前的 AST 完全一致。

接口依据是 simcode10 归档的
[`六模型报告`](../simcode10/pingpong_moz1-main/pingpong_controller/tools/rl_2real/FPLC_MODEL_VERSIONS_AND_CURRICULA_20261026.md)
及配套 `FPLC_SELECTED_MODEL_CONTRACTS_20261026.json`。
归档没有携带 V224 等新训练分支的完整冻结源码，因此 V224 的几何 oracle 按归档公式独立计算；
这不等同于新训练环境的 MJX/JIT 全链路对拍或真实机械臂随动验收。
