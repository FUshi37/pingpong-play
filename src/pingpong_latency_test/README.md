# Pingpong Latency Test Package

测试乒乓球检测系统从图像捕捉到输出检测状态的延迟。

## 测试原理（单时钟法，推荐）

笔记本屏幕翻白的瞬间，向机器人发一个 UDP 包。机器人用**自己的时钟**同时记录两个时刻：
UDP 到达时刻 `arrival_time` 和相机检测到白屏的时刻 `detect_time`。

```
笔记本翻白 ──UDP包──> 机器人记 arrival_time (机器人钟)
         └─屏幕发光─> 相机曝光->传输->解码->检测 ──> detect_time (机器人钟)

端到端延迟 = detect_time - arrival_time - 网络单程延迟
```

因为两个时刻都在**同一个机器人时钟**上，相减**不需要任何跨机器时钟同步**，
彻底绕开了笔记本时钟漂移 / Windows NTP 阶跃的问题（这是之前 293ms 误差的根源）。

残留误差只有：网络单程延迟（~1ms，可用 clock_sync 实测扣除）+ 帧量化（60fps→16ms，多次平均后趋于0）。
注意：测得的"屏幕+曝光+相机"段里还含 LCD 屏幕翻白→发光的延迟（真实乒乓球场景没有屏幕），
所以这段是**上界**。

## 组件

机器人端（ROS 包）：
- **flash_detector**：检测黑→白转换 + UDP 监听，记录到 `flash_events.json` 和 `flash_events_udp.json`
- **roi_calibrator**：交互式框选屏幕区域、标定亮度阈值
- **latency_analyzer**：配对计算延迟
- **clock_sync**：测网络单程延迟（取 min-RTT/2）

笔记本端（`laptop/` 目录，纯标准库，拷到笔记本用 python 直接跑）：
- **flash_generator.py**：全屏黑白频闪 + 翻白时发 UDP 包
- **clock_sync.py**：clock_sync 的另一端

## 编译

```bash
cd /home/spirit-ai/pingpong_moz1
colcon build --packages-select pingpong_latency_test
source install/setup.bash
```

## 完整流程

### 1. 标定 ROI 和阈值（一次即可）

```bash
ros2 run pingpong_latency_test roi_calibrator --ros-args \
    -p image_topic:=/camera/camera/color/image_raw/compressed
```
鼠标框住屏幕 → 常亮白屏按 `w` → 常亮黑屏按 `b` → 按 `p` 打印参数。

### 2. 测网络单程延迟（一次即可，记下 min-RTT/2）

由于 Windows 入站防火墙常拦 UDP，让**机器人当 server**（Linux 无防火墙），笔记本当 client：

```bash
# 机器人端
ros2 run pingpong_latency_test clock_sync --server --port 5005
```
```bash
# 笔记本端
python clock_sync.py --client <机器人IP> --port 5005 --count 500
```
记下打印的 `est. one-way net delay (min-RTT/2)`，例如 0.99ms → 后面用 `--net-delay 0.001`。

### 3. 跑延迟实验

```bash
# 机器人端（用步骤1标定的参数，udp_port 默认 5006）
ros2 run pingpong_latency_test flash_detector --ros-args \
    -p image_topic:=/camera/camera/color/image_raw/compressed \
    -p roi:=354,197,134,74 \
    -p brightness_threshold:=0.624 \
    -p transition_hysteresis:=0.248 \
    -p udp_port:=5006 \
    -p log_file:=flash_events.json
```
机器人端应打印 `UDP trigger listener: port 5006`，收到第一个包时打印 `UDP trigger: first packet ...`。

```bash
# 笔记本端
python flash_generator.py --robot-ip <机器人IP> --udp-port 5006 \
    --interval 2.5 --count 30
```
全屏黑白闪烁，每次翻白发 UDP 包。跑完 30 次自动退出（或按 q/ESC）。

### 4. 分析

```bash
ros2 run pingpong_latency_test latency_analyzer flash_events.json \
    --udp flash_events_udp.json --net-delay 0.001
```

输出端到端延迟（屏幕→检测状态）的统计，以及软件侧分段（transport/decode/detection）。

## 关键参数

flash_detector:
- `image_topic` / `roi` / `brightness_threshold` / `transition_hysteresis`：见 roi_calibrator
- `udp_port`：UDP 触发端口（默认 5006，设 0 关闭 UDP，回退到纯检测）
- `log_file`：检测事件输出（UDP 到达自动存到 `<stem>_udp.json`）

flash_generator.py (laptop):
- `--robot-ip`：机器人 IP（必填）
- `--udp-port`：要和 flash_detector 的 udp_port 一致
- `--interval`：翻白间隔秒（默认 2.5，要 > min_flash_interval 0.3）
- `--white-ms`：白屏持续 ms（默认 600）
- `--count`：白屏次数（默认 30，0=不停）
- `--warmup`：预热次数，不发不记（默认 2）

latency_analyzer:
- `--udp <file>`：启用单时钟法（推荐）
- `--net-delay`：网络单程延迟秒（用 clock_sync 的 min-RTT/2）
- `--laptop <csv/json>` + `--clock-offset`：旧的跨时钟法（不推荐，受时钟漂移影响）

## UDP 连通性排查

机器人当 server 收不到笔记本的包时：
1. 笔记本能 ping 通机器人？（同网段）
2. 机器人 `flash_detector` 是否打印了 `UDP trigger listener: port ...`？
3. 笔记本发包目标 IP/端口对不对？（`--robot-ip`、`--udp-port`）
4. 机器人 Linux 防火墙（一般 ufw 不活动，不拦）

注意：UDP 触发方向是**笔记本→机器人**（出站），Windows 默认放行出站，所以这个
方向通常没有防火墙问题（和 clock_sync 让机器人当 server 是同一个道理）。

## Sim2Real 应用

端到端延迟（扣掉屏幕上界后）可用于：策略前向预测补偿球位、仿真复刻同等延迟、
调 Kalman/OneEuro 参数。真实 YOLO 球检测还要再叠加推理耗时（`ball_detector.py` 的 `last_yolo_ms`）。

## 测 D+E：YOLO 推理 + 后处理（pipeline_profiler）

频闪法测的是 A+B+C（曝光+传输+解码，含屏幕上界），**不经过 YOLO**。`pipeline_profiler`
补上剩下两段，且**直接 import 真实的 `PingPongBallDetector`**、按 `pingpong_node` 的方式
构建，所以测出来就是你打球时跑的同一套代码：

- **C decode**：`cv2.imdecode`（这里也测一遍，应与频闪的 decode ~1.5ms 吻合）
- **D yolo**：`detector.last_yolo_ms`，YOLO 前向
- **E postprocess**：`get_observation` 总耗时减去 yolo = 几何测距+圆拟合+Kalman/OneEuro+坐标变换

完整「光子→球状态」延迟 ≈ **频闪端到端(A+B+C+屏幕) + (D + E)**。
注意频闪的 A+B+C 里已含一次 decode，所以诚实的合成是：
`photons->state ≈ flash_e2e + yolo_ms + postprocess_ms`（用频闪的 decode 代表 C，不重复计）。

```bash
# 在 (pingpong) conda 环境里跑（YOLO/ultralytics 在那儿）
source install/setup.bash
ros2 run pingpong_latency_test pipeline_profiler --ros-args \
    -p image_topic:=/camera/camera/color/image_raw/compressed \
    -p camera_info_topic:=/camera/camera/color/camera_info \
    -p yolo_model_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/models/best.pt \
    -p infer_imgsz:=640 -p device:=0 -p use_kalman:=true \
    -p out_csv:=pipeline_latency.csv
```

**在相机视野里挥动乒乓球**让 YOLO 真的检到目标——检到的帧才会跑完整的 E（几何+滤波），
统计按 detected/missed 分开打印，以 **detected 帧的 D+E total** 为准。Ctrl+C 结束，
存 `pipeline_latency.csv`。关键参数要和打球时一致：`infer_imgsz` / `device` / `half` /
`use_kalman`（默认按 `pingpong_node` 取值）。也可以 `ros2 bag play` 录好的包来离线测。

最终延迟数值21-58ms，理想范围30-40ms。