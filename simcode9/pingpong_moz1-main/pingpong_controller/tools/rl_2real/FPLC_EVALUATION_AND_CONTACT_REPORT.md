# FPLC 选定模型 validate 与实物验收报告

此报告保存所选四模型的 validate 依据和实物离线检查要求。模型是否可部署由接口配对和监督硬件验证决定；本次不提交历史碰撞搜索、可达性诊断或批量实验记录。

## 方法代码

共用 `validate_juggle_mjx_ppo.py` 和 `screen_juggle_mjx_checkpoints.py` 随第一批训练／validate 提交，用于 checkpoint 加载、配置复原、确定性或随机回放、批量筛选与动作／接触指标记录。对应环境、PPO、观测和奖励辅助模块必须同时存在。按用户要求不提交 unit tests 或其他回归测试源码。

碰撞 feature `ball_racket_contact_real11_static_1ms_v1` 的 XML、编译后几何修正与精确参数随第一批仿真提交保存，依据和限制见 `BALL_RACKET_CONTACT_REAL11_STATIC_1MS_V1_REPORT.md`。本次不提交旧 fitter、临时脚本复制件或历史标定数据包。

V152 与 V151 accepted release 的已有确定性回放均完成 1200 steps、13 hits，十次 steady apex 均在 1.40–1.45 m；对应 `video_evidence.json` 与命名 release 视频随模型保存。这是仿真证据。V98 real11 的静态首弹结果不能替代动态实物碰撞验收。

## 选定 clean 后继的证据

| 指标与原 CSV 字段 | GPU0 V161 S7 u331 | GPU1 V172 S10 u314 |
| --- | ---: | ---: |
| rolling `recent_mean_len`，steps | 1200 | 1200 |
| rolling `recent_episode_truncation_rate` | 1.0 | 1.0 |
| rolling `recent_rapid_steady_late_in_band_rate` | 0.999124 | 0.996597 |
| rolling steady racket XY RMS，m/s | 0.046947 | 0.047301 |
| rolling steady racket omega RMS，rad/s | 0.221327 | 0.254165 |
| rolling mean contact center distance，m | 0.002654 | 0.004750 |
| rolling late adjacent apex jump，m | 0.007001 | 0.008353 |

这两行分别同时通过 rolling convergence、advance 和 clean retention；精确完整字段随 `FPLC_CLEAN_STAGE_ACCEPTANCE_20261010.json` 保存。GPU0 S8、GPU1 S11 开始观测 DR，两个目录的通用 `mjx_curriculum_best.pkl` 已属于 DR，必须排除。选择依据是完整 clean 晋级证据，不是对所有历史 actor 做过新的同种子全面排名。

## 实物数据与验收

在 `policy_joint_sample.py` 和 ball handover 接口记录原始 q／dq、原始相机帧和源时间戳、policy tick、frame age、预测／原始状态、FK site／surface、actor 向量、原动作／投影后动作、q_ref、安全介入和实际发布目标。不得用上一拍指令充当测量状态，或用插值未来帧构造 actor。

观测配对先覆盖 72-D／156-D 向量和历史初始化，再覆盖 actor 确定性输出、连续多 tick q_ref 积分、投影介入／回退／reset。记录模型、配置、坐标标定和源码哈希。在球／拍面固定的 paired A/B 中单独检查 raw sensor depth 与 surface-to-center depth；3 mm 仿真 geom offset 不能用于统一补偿视觉。

颠球统计至少包含物理 confirmed hit、P(N≥1/3/12)、回合长／truncation、第四拍以后绝对 apex 与相邻跳变、ball／racket XY 速度、完整角速度、接触偏心、视野、qvel／qacc 和失败分布。模拟真实 hit_count、contact context 的可用性必须在部署报告中解决；缺少真实可用事件不能以仿真真值填充。碰撞时刻附近保留输出速度、pose／twist 与数据有效性，不把 return 或单段视频当作硬件验收。
