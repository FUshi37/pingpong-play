# 相机与动捕球数据同时录制

运行 `RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 后，在浏览器中点击 **Record** 开始录制，点击 **Stop** 保存。脚本默认开启相机和动捕；在录制期间，两路数据均独立读取并保存，不受界面选择的 `ballSource` 影响。控制使用 camera、mocap 或 drag 时都能录制两路传感器数据。

输出目录为：

```text
src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_YYYYMMDD_HHMMSS/
```

可通过 `PROBE_RECORDING_OUTPUT_ROOT` 更改输出根目录。

## 长时间录制：选择是否保存相机图像

浏览器 **Record** 旁新增 **保存相机图像** 复选框，录制前选择，录制期间不可修改。默认勾选，保持现有行为。取消勾选后，相机 JPG 与原始 IR/深度/RGB 均不保存；相机仍照常检测，球位置、速度、双路传感器数据和控制日志继续保存。

长时间测试可以用以下任一方式将默认值设为关闭，再按需在浏览器勾选：

```bash
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh --no-save-camera-images
# 或
PROBE_RECORDING_SAVE_CAMERA_IMAGES=false ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

`--save-camera-images` 可恢复默认开启。`ENABLE_RAW_CAMERA_RECORDING` 仍控制原始帧录制能力；只有该能力已开启且本次勾选保存图像时才会写入原始帧。原有 `PROBE_RECORDING_IMAGE_RATE_HZ=0` 只关闭可视化 JPG，不能代替同时关闭所有相机图像的新开关。选择结果写入 `metadata.json` 的 `recording.saveCameraImages` 和 `summary.json` 的 `saveCameraImages`。

## 长时间录制：仅记录球轨迹

浏览器 **Record** 旁的 **仅记录球轨迹** 开关默认关闭，录制前选择，录制期间锁定。
开启后不保存策略 obs、动作、关节状态、命令、仿真记录、界面快照或浏览器历史。
仍按原录制频率保存两路传感器的球位置、时间戳、有效性，以及原始/控制/预测球位置。
为了统计真实颠球次数，还保留少量实测球速、球拍 XYZ 和时钟/标记身份；不保存关节向量。
图像开关独立生效。长时间无图像录制可以直接启动：

```bash
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh --recording-trajectory-only --no-save-camera-images
# 或
PROBE_RECORDING_TRAJECTORY_ONLY=true PROBE_RECORDING_SAVE_CAMERA_IMAGES=false \
  ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

`--no-recording-trajectory-only` 恢复完整录制默认值。每次录制的选择分别写入
`metadata.json` 的 `recording.trajectoryOnly` 和 `summary.json` 的 `trajectoryOnly`，不改变下一次启动的默认配置。

精简模式的数值数据只写入 `control_samples.jsonl`（球字段与精简 `ballSensors`），
不再另写重复的 `ball_sensor_samples.jsonl`；事件保存在 `events.jsonl`，次数保存在 `summary.json.juggle`。
不勾选保存图像时，只产生这两个逐行数据文件及 metadata/summary。
次数在写入进程中增量计算，页面仍显示 **颠球 N 次**，不依赖图像或离线分析。

停止后自动执行轨迹分析，**Full analysis** 在该模式下禁用。分析逐行读取、逐行导出
`analysis/ball_trajectory.csv`，CSV 保留全部采样的球位置、两路原始传感器位置与时间。
曲线图和网页回放最多保留 8000 个等间隔样本，3D 回放和图像索引最多各 3000 条，
不把完整日志加载进内存；完整 JSONL 和 CSV 不做降采样，颠球计数也不用降采样结果。
可生成球位置/速度曲线与网页轨迹回放，不生成策略、关节、仿真对比或完整视频分析。
长时间缩略曲线可能看不清单次击球，精细检查应读取对应时间段的原始记录。

修改启动脚本后需要重启探针才能使用新选项；正在运行的旧进程不会自动切换录制格式。

2026-10-29 离线验证：315 项录制、界面、模型及部署回归通过。
从 `probe_recording_20261029_201036` 逐行读取前 10000 条控制记录，
精简前后颠球计数均为 26，控制日志体积减少 93.13%；完整数值数据未降采样。
测试范围与数据见 [`trajectory_only_20261029.json`](../config/policy_validation_reports/trajectory_only_20261029.json)。

## 录制结束后的颠球次数

点击 **Stop**，日志写入完成后，页面状态显示 **颠球 N 次**，并将统计保存在录制目录 `summary.json` 的 `juggle` 中。每次识别的击球还会在 `events.jsonl` 中写入一条 `event="juggle_bounce"`，包含序号、时间、数据源和击球前后的竖直速度。计数不依赖图像保存或 **Full analysis**。

计数使用当前 `ballSource` 选中的相机或动捕有效观测，同一次击球不会把两路传感器相加；`drag` 不参与真实颠球计数。判据为 90 ms 内竖直速度从低于 −0.10 m/s 转为高于 +0.10 m/s，速度增量至少 0.70 m/s，且击球前或后的球位于球拍附近（水平距离不超过 95 mm，竖直间隙 −35 至 120 mm）。两次击球至少间隔 160 ms，与现有离线轨迹分析的判据保持一致。

重复传感器时间戳只处理一次；丢球、无效观测、超过 90 ms 的观测间隙、传感器/标记切换及同步时钟代次变化会隔断判定窗口。预测球状态不用于计数。统计在低优先级录制进程中随日志增量完成，最多保留 32 条近期观测，不随录制时长累积轨迹到内存。

`juggle.count` 为次数，`bySource` 给出各选中数据源贡献的次数，`measurementSamples` 为有效去重观测数。`status="no_valid_observations"` 表示没有可用观测，不能将零次理解为有效测试结果；`status="partial"` 表示录制队列有丢弃，次数可能不完整。该统计为基于观测的估计，传感器漏检或实际击球超出判据范围可能漏计。

## 双路球数据格式

新增 `ball_sensor_samples.jsonl`，每行对应一个已保存的控制采样，默认 200 Hz；相同数据也位于 `control_samples.jsonl` 的 `ballSensors` 字段中。两份数据复用同一条异步写入队列消息，文件写入由现有录制进程完成。

为保护 200 Hz 控制周期，传感器产生数据时就准备好不可变的录制缓存。控制线程只取两路缓存的引用及采样时刻，不增加传感器轮询、传感器锁、数组转换、诊断格式化或文件写入。录制进程按**控制采样时刻**检查缓存的超时、时钟重置和有效性，再生成两份日志；磁盘写入延迟不会改变记录的数据年龄。默认相机和动捕均运行在独立进程中；诊断用的非子进程模式由原有相机/UDP 接收线程准备缓存。录制队列沿用非阻塞写入，队列满时丢弃录制消息并统计 `dropped`，不等待磁盘或降低控制频率。

| 字段 | 含义 |
| --- | --- |
| `sourceSeq`, `sourceTimeS`, `tick`, `time`, `wallTimeS` | 与控制日志对应的序号和时间 |
| `ballSource` | 控制当前使用的数据源，不决定录制哪些传感器 |
| `receiverTimeS` | 读取缓存时的公共 ROS 时钟时间，秒 |
| `camera`, `mocap` | 各自独立的球数据及状态 |
| `bothValid` | 当前两路观测都有效 |
| `comparisonReady` | 两路均有效、时钟已同步、坐标系已对齐 |
| `measurementTimeDeltaS` | 相机采样时间减去动捕采样时间，秒；缺少时间时为 `null` |

每路数据包含：

- `enabled`, `available`, `valid`：是否启用、是否有读取接口、当前观测是否有效。
- `position`, `velocity`：传感器输出的位置（米）与速度（米/秒），采用现有转换后的 `sim_frame` 坐标；不加入 probe 的人工噪声、保持值或遮挡预测。传感器自身的检测和平滑参数仍然生效。
- `controlFrame`, `detectorFrame`, `frameAligned`：坐标系及标定对齐状态；相机未完成坐标标定时 `comparisonReady=false`。
- `measurementTimestampS`：现有时钟同步逻辑映射到 ROS 时钟的采样时间，保留完整浮点精度。
- `sourceTimestampS`, `receiverTimestampS`, `measurementAgeS`, `clockSync`：原始来源时间、接收时间、数据年龄和时钟同步诊断。原始来源时间可能属于不同设备时钟，不能直接跨设备相减。
- `measurement`：相机检测诊断（含 `observation_state`、深度和检测方法等）或动捕诊断（含 `frameIdx`、`markerId`、`tracked`、包计数和标记选择信息）。
- `error`：读取发生异常时的错误信息。

动捕暂无数据、丢球、超时、未同步或未启用时，其 `valid=false`，`position` 和 `velocity` 为 `null`；相机仍继续记录。相机失效时同样保留动捕记录。不会使用零坐标、控制保持值或预测值冒充有效观测。

记录的是每个控制采样时刻各传感器的**最新缓存观测**。相机与动捕采样频率不同，相同时间戳可能连续出现；这不表示传感器产生了新观测，也不保证保留每个原始传感器帧。后续分析应依据各自 `measurementTimestampS` 和 `clockSync` 的时钟代次去重，按公共采样时间插值或匹配，再计算位置与速度误差；不要直接将同一行视为同时采集。时钟切换或重置期间避免跨代次匹配。

简单读取示例：

```python
import json
from pathlib import Path

run_dir = Path("实际录制目录")
with (run_dir / "ball_sensor_samples.jsonl").open() as f:
    rows = [json.loads(line) for line in f]

pairs = [row for row in rows if row["comparisonReady"]]
for row in pairs[:5]:
    print(row["sourceSeq"], row["measurementTimeDeltaS"],
          row["camera"]["position"], row["mocap"]["position"])
```

`summary.json` 中的 `counts.ballSensors` 是已保存的双路记录行数；`dropped` 是录制队列丢弃的消息总数。数据格式版本为 12，旧控制字段保持兼容。

点击 **Stop** 并完成日志写入后，会在低优先级的独立分析进程中自动生成以下文件，浏览器分析页新增 **Camera / Mocap** 入口：

- `analysis/ball_sensor_position_comparison.png`：X/Y/Z 位置对比，各自采用实际采样时间。
- `analysis/ball_sensor_position_difference.png`：相机减去动捕的 X/Y/Z 差值（毫米）与三维差值大小。
- `analysis/ball_sensor_position_difference.csv`：按相机采样时间线性插值对齐的动捕位置及差值。
- `analysis/ball_sensor_positions.csv`：两路去重后的有效采样及原始时间戳。
- `analysis/ball_sensor_comparison_report.json`：有效数据比例、采样数量、偏差、RMSE 和差值分位数。

两路位置对比图保留无效区间的断线。差值仅使用已同步、已完成坐标对齐的数据；不外推，也不跨越动捕丢失、时钟重置或超过 50 ms 的采样缺口进行插值。统计针对独立相机采样，避免重复控制缓存对统计结果加权。现有时钟同步未标定固定传输延迟，所以这些差值仍可能含时间偏差，不能直接等同于纯相机空间测量误差。

浏览器的 **Full analysis** 复选框控制视频和策略等完整分析；取消勾选仍会绘制球位置对比与差值图。绘图在录制结束后执行，未向 200 Hz 控制循环添加绘图或文件读取工作。正在运行的旧脚本需要下次重新启动后才使用新增分析逻辑。
