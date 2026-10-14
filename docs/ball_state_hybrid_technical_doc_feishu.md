# YOLO + 红外反光混合检测与 200Hz 球状态估计技术文档

| 文档项 | 内容 |
| --- | --- |
| 适用工程 | `pingpong_moz1` |
| 适用节点 | `pingpong_controller/pingpong_node.py` |
| 检测方案 | YOLO + IR 反光混合检测 |
| 状态估计方案 | `BallStateEstimator`，60Hz 视觉更新 + 200Hz 控制预测 |
| 输出坐标系 | 优先 `base_link`，未加载手眼标定时退回 `camera` |
| 主要调试话题 | `/pingpong/ball_state` |
| 主要日志 | `outputs/rl_sim/estimator_log_*.csv` |

## 1. 背景与目标

当前实物乒乓球颠球系统使用 RealSense 采集 RGB、IR、Depth 三路数据。灰色反光球在普通 RGB 图像中颜色特征不稳定，但在 IR 图像中反光亮斑明显。因此系统采用“RGB YOLO 语义粗定位 + IR 亮斑精定位 + Depth/Geometry 深度融合”的混合检测方案。

控制侧以 200Hz 运行，而相机检测通常为 60Hz。仅使用最新检测值会产生三个问题：

1. 控制周期中间没有新测量，球状态呈阶梯状。
2. 检测完成时球已经运动了一段时间，存在感知延迟。
3. 球上抛飞出 FOV 时，短时间内没有真实检测，策略会长时间看到过期状态。

为解决这些问题，当前代码引入 `BallStateEstimator`，统一完成：

| 目标 | 当前方法 |
| --- | --- |
| 60Hz 到 200Hz 频率桥接 | 每个 200Hz 控制 tick 调用 `predict_to(t_now)` |
| 感知延迟补偿 | 使用图像采集时刻 `t_img` 更新滤波器，再预测到当前时刻 |
| 短时丢失鲁棒性 | 在可信时间窗内按弹道模型外推，超时后置为 invalid |
| 深度噪声抑制 | 测量噪声按横向和深度轴各向异性建模 |
| 击打/反弹适应 | 创新门控、连续拒绝逃逸重初始化、事件过程噪声放大 |

注意：当前运行命令同时设置 `enable_ball_state_estimator:=true` 和 `enable_fov_loss_compensation:=true`。代码运行时会自动关闭 FOV-loss compensator，让 `BallStateEstimator` 独占丢失期间的预测逻辑，避免同一段轨迹被弹道传播两次。

## 2. 总体架构

### 2.1 在线数据流

| 阶段 | 输入 | 处理模块 | 输出 |
| --- | --- | --- | --- |
| 1. 采集 | RealSense RGB / IR / Depth | `HybridBallDetector` 管理相机流 | 同步帧组 |
| 2. 粗定位 | RGB 图像 | YOLO `best.pt` | 球的 bbox ROI |
| 3. 精定位 | IR 图像 + YOLO ROI | IR 阈值、形态学、椭圆拟合、像素 KF | IR 像素中心和半径 |
| 4. 深度估计 | Depth 图 + IR 半径 | sensor / geometry 均可计算，但近距离颠球不强依赖硬件 depth | `depth_mm` |
| 5. 三维重建 | IR 像素 + 深度 + 内外参 | IR 反投影，IR 到 RGB，RGB 到 base_link | 3D 位置 |
| 6. 视觉更新 | 60Hz 真实检测 | `BallStateEstimator.update()` | 滤波器内部状态 |
| 7. 控制预测 | 200Hz 控制 tick | `BallStateEstimator.predict_to()` | 当前时刻球状态 |
| 8. 策略输入 | 球位置/速度/age | RL 控制或调试发布 | 关节命令或 `/pingpong/ball_state` |

### 2.2 关键模块

| 文件 | 职责 |
| --- | --- |
| `hybrid_ball_detector.py` | YOLO + IR + 深度融合检测核心，支持在线和离线复现 |
| `hybrid_ball_detector_adapter.py` | 将混合检测输出适配为 `PingPongBallDetector` 兼容格式 |
| `ball_state_estimator.py` | 位置测量融合、弹道预测、感知延迟补偿、丢失时间线 |
| `pingpong_node.py` | ROS2 参数、线程、日志、控制循环、发布与记录 |
| `annotate_raw_video_hybrid.py` | 使用录制的 RGB/IR/Depth/camera_params 离线复现混合检测并生成标注视频 |

## 3. 混合检测方法

### 3.1 YOLO 语义粗定位

RGB YOLO 的职责不是提供最终像素中心，而是提供语义过滤和 ROI。流程如下：

1. 每 `yolo_interval` 帧运行一次 YOLO。
2. 仅接受 `target_class` 对应类别，当前命令为 `Class 0`。
3. 选择置信度最高的目标框。
4. 对 bbox 外扩 `hybrid_roi_margin` 像素，形成 IR 搜索 ROI。
5. 若 YOLO 连续丢失小于 `max_yolo_miss=3`，继续使用最近 bbox。
6. 若 ROI 内 IR 检测失败，立即退回全图 IR 检测。

该设计把 YOLO 作为“软约束”，避免 YOLO 单帧漏检导致整帧失效。

### 3.2 IR 反光球精定位

IR 图像中反光球通常表现为高亮连通域。检测流程：

| 步骤 | 方法 |
| --- | --- |
| 阈值 | 固定阈值或自适应阈值，参数来自 `ir_params.json` |
| ROI 限制 | 优先在 YOLO ROI 内二值化搜索 |
| 形态学 | 膨胀、闭运算、开运算去噪 |
| 候选提取 | 查找外轮廓，过滤小面积噪声 |
| 形状过滤 | 半径范围、长短轴比、面积圆度 |
| 中心估计 | 椭圆中心 70% + 矩重心 30% |
| 候选选择 | 选择总亮度最高的候选 |
| 像素平滑 | `PixelKalmanFilter` 平滑 IR 像素中心 |

IR 检测输出的核心字段：

| 字段 | 含义 |
| --- | --- |
| `cx`, `cy` | 卡尔曼平滑后的 IR 像素中心 |
| `cx_raw`, `cy_raw` | 平滑前的原始 IR 检测中心 |
| `radius` | 椭圆短轴半径，单位 px |
| `aspect_ratio` | 椭圆短轴与长轴比例 |
| `brightness` | 候选区域亮度和 |
| `detection_method` | `hybrid` 或 `ir_fallback` |

### 3.3 实测约束与方案取舍

实物颠球和静态/慢速目标不同，当前场景有几个关键约束：

| 约束 | 影响 | 工程取舍 |
| --- | --- | --- |
| 颠球距离较近 | RealSense D455 深度在近距离容易空洞、跳变或落在工作距离边界附近 | 硬件 depth 不作为主可信来源，更多用于记录和对照 |
| 球速较快 | RGB 图像中球会拖影，外观常呈椭圆而不是稳定圆 | 不能简单把逐帧椭圆半径变化都当作真实深度变化 |
| YOLO 依赖 RGB 外观 | 低曝光下 RGB 图像过暗，YOLO 语义检测可靠性下降 | YOLO 只作为辅助语义约束，不能作为低曝光主检测源 |
| IR 支持低曝光 | 可显著压低运动拖影，反光球中心更稳定 | IR 是当前实物颠球的主定位来源 |

因此，文档中的 `sensor` / `geometry` 深度选择应理解为“检测器具备的两条深度候选路径”，而不是“实物颠球时硬件 depth 一定更好”。在近距离高速颠球中，主要可信信息是低曝光 IR 的二维中心；深度和速度需要通过稳定化、状态估计和弹道约束来降低噪声。

### 3.4 深度估计与选择

系统同时计算两类深度：

| 深度源 | 适用场景 | 说明 |
| --- | --- | --- |
| `sensor` | 中远距离、深度图有效时 | RealSense depth 图在球区域采样，中位数聚合；近距离颠球时效果可能更差 |
| `geometry` | 低曝光 IR 半径较稳定时 | 由球半径和相机焦距单目估距；高速拖影导致半径不稳定时也会退化 |

单目几何深度公式：

$$Z_{geom}=\frac{f_{ir}D}{2r}$$

其中：

| 符号 | 含义 |
| --- | --- |
| $$Z_{geom}$$ | 几何深度，单位 mm |
| $$f_{ir}$$ | IR 相机平均焦距，单位 px |
| $$D$$ | 乒乓球直径，默认 40mm |
| $$r$$ | IR 图中球的像素半径 |

硬件深度从 depth 图中读取。若球 mask 内有效深度点不少于 5 个，则优先用 mask 区域；否则用 `depth_neighborhood_px` 指定的邻域。当前命令为 `depth_neighborhood_px:=2`，即中心周围 5x5 区域。实测近距离颠球时，硬件深度可能比几何深度更差；如果日志中 `depth_method=sensor` 伴随明显跳变，应避免强行使用 sensor 深度。

深度选择使用带迟滞的距离门：

$$Z=
\begin{cases}
Z_{geom}, & \hat{Z}<500mm \\
Z_{sensor}, & \hat{Z}>650mm \\
\text{保持上一深度源}, & 500mm \leq \hat{Z} \leq 650mm
\end{cases}$$

选择后再进行时域去尖峰和平滑。若历史深度中位数为 $$\tilde{Z}$$，当前原始深度为 $$Z_{raw}$$，则软钳制门限为：

$$G=\max(150mm,0.25\tilde{Z})$$

当 $$|Z_{raw}-\tilde{Z}|>G$$ 时，将深度限制到历史中位数附近，再进入长度为 5 的中位数平滑窗口。

需要注意：高速运动下球像呈椭圆时，半径 $$r$$ 同时混入真实距离、运动拖影、曝光、阈值和形态学膨胀的影响。因此，几何深度更适合作为可用但有噪声的观测量，不应把单帧 $$Z_{geom}$$ 的跳变直接解释为球的真实深度跳变。

### 3.5 三维重建与坐标变换

IR 像素反投影到 IR 相机坐标系：

$$X_{ir}=\frac{(u_{ir}-c_{x,ir})Z}{f_{x,ir}}$$

$$Y_{ir}=\frac{(v_{ir}-c_{y,ir})Z}{f_{y,ir}}$$

$$P_{ir}=\begin{bmatrix}X_{ir}&Y_{ir}&Z\end{bmatrix}^T$$

由于手眼标定基于 RGB 相机，系统先用 RealSense 的 IR 到 RGB 外参将点变换到 RGB 相机坐标系：

$$P_{rgb}=T_{rgb\leftarrow ir}P_{ir}$$

若加载了手眼标定，则继续转换到机器人 `base_link`：

$$P_{base}=T_{base\leftarrow rgb}P_{rgb}$$

最终混合检测输出：

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `position` | mm | 目标坐标系下的球位置 |
| `velocity` | mm/s | 由最近 5 帧位置历史差分得到的速度 |
| `raw_measurement.depth_mm` | mm | 本帧使用的稳定深度 |
| `raw_measurement.depth_method` | 字符串 | `sensor` 或 `geometry` |
| `raw_measurement.raw_position_ir` | mm | IR 相机系原始三维点 |
| `raw_measurement.raw_position_rgb` | mm | RGB 相机系三维点 |

## 4. 球状态估计方法

### 4.1 输入与输出

`BallStateEstimator` 是一个纯 NumPy 状态估计器，不依赖 ROS。它由两条线程调用：

| 调用位置 | 频率 | 方法 | 作用 |
| --- | --- | --- | --- |
| 视觉线程 | 约 60Hz | `update(position_mm, t_img)` | 融合真实检测 |
| 控制线程 | 200Hz | `predict_to(t_now)` | 输出当前控制时刻球状态 |

状态向量定义：

$$x=\begin{bmatrix}p_x&p_y&p_z&v_x&v_y&v_z\end{bmatrix}^T$$

观测向量为三维位置：

$$z=\begin{bmatrix}p_x&p_y&p_z\end{bmatrix}^T$$

观测矩阵：

$$H=\begin{bmatrix}I_{3\times3}&0_{3\times3}\end{bmatrix}$$

输出包含：

| 字段 | 含义 |
| --- | --- |
| `position` | 预测到 `t_now` 的位置，单位 mm |
| `velocity` | 预测到 `t_now` 的速度，单位 mm/s |
| `valid` | 当前球状态是否可用于控制 |
| `age_s` | 传给策略的观测年龄 |
| `state` | `tracking` / `predicting` / `lost` |
| `loss_s` | 距离最近真实测量的时间 |
| `extrapolation_s` | 本次从滤波器时间推进到当前时刻的时长 |
| `clamped` | 是否超过外推上限并被截断 |

### 4.2 时间戳与感知延迟补偿

真实检测以图像采集时间 $$t_{img}$$ 更新滤波器，而不是以检测完成时间更新。控制线程在当前时刻 $$t_{now}$$ 调用预测：

$$\Delta t=t_{now}-t_{img}$$

因此，只要 $$t_{img}$$ 和 $$t_{now}$$ 在同一时钟域中，预测过程天然补偿从曝光、传输、YOLO/IR 检测到控制 tick 的感知延迟。

当前实现只补偿感知延迟，不补偿执行延迟。执行延迟已在 RL 训练时通过执行器模型处理，因此 `ball_estimator_prediction_horizon_s` 默认保持 0。

### 4.3 运动模型

基础弹道模型：

$$p(t+\Delta t)=p(t)+v(t)\Delta t+\frac{1}{2}g\Delta t^2$$

$$v(t+\Delta t)=v(t)+g\Delta t$$

当前默认启用空气阻力，动力学为：

$$\dot{p}=v$$

$$\dot{v}=g-k_d\|v\|v+k_m(\omega\times v)$$

其中：

| 符号 | 当前含义 |
| --- | --- |
| $$g$$ | 重力向量，`base_link` 中为 $$[0,0,-9810]^Tmm/s^2$$ |
| $$k_d$$ | 二次空气阻力系数，默认 `1.117e-4 / mm` |
| $$k_m$$ | Magnus 系数，默认 0 |
| $$\omega$$ | 已知自旋角速度，当前无外部自旋传感，默认 0 |

启用空气阻力或 Magnus 时，系统使用 RK4 按 `aero_substep_s=0.005s` 子步积分。

### 4.4 Kalman 预测与更新

预测：

$$x_k^-=f(x_{k-1},\Delta t)$$

$$P_k^-=F_kP_{k-1}F_k^T+Q_k$$

更新：

$$y_k=z_k-Hx_k^-$$

$$S_k=HP_k^-H^T+R$$

$$K_k=P_k^-H^TS_k^{-1}$$

$$x_k=x_k^-+K_ky_k$$

$$P_k=(I-K_kH)P_k^-$$

系统用前两帧真实检测做 two-point 初始化，直接估计初始速度：

$$v_0=\frac{p_1-p_0}{t_1-t_0}$$

这样比从零速度启动更适合高速乒乓球。

### 4.5 测量噪声建模

检测位置噪声不是各向同性的。像素横向中心通常较准，而深度轴噪声更大。代码先在相机系构造测量协方差：

$$R_{cam}=\operatorname{diag}(\sigma_{lat}^2,\sigma_{lat}^2,\sigma_{depth}^2)$$

当前默认：

$$\sigma_{lat}=20mm$$

$$\sigma_{depth}=60mm$$

若输出为 `base_link` 且存在 `R_base_camera`，协方差旋转到估计器坐标系：

$$R_{est}=R_{base\leftarrow camera}R_{cam}R_{base\leftarrow camera}^T$$

过程噪声同样按相机深度轴做各向异性缩放。当前节点参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `ball_estimator_q_depth_pos_mult` | 0.50 | 深度轴位置过程噪声缩放 |
| `ball_estimator_q_depth_vel_mult` | 0.20 | 深度轴速度过程噪声缩放 |

这能抑制 D455 深度抖动被误解释成高速速度跳变。

### 4.6 创新门控与事件重初始化

每个真实测量进入滤波器前，会检查创新：

$$e=z-p^-$$

位置跳变门限：

$$\|e\| \leq 250mm$$

深度轴跳变门限：

$$|e_{depth}| \leq 180mm$$

若连续拒绝达到 `gate_escape_rejects=3`，认为可能发生真实击打或反弹，不再永久拒绝，而是用这些被拒绝样本重新拟合速度并重初始化。

反弹/击打事件还会临时放大过程噪声：

$$Q_{event}=50Q$$

持续 `event_boost_ticks=4` 个更新 tick，使滤波器更快贴合接触后的新轨迹。

### 4.7 丢失时间线

`predict_to()` 根据距离最近真实测量的时间 $$loss_s$$ 划分状态：

| 时间范围 | 状态 | 行为 |
| --- | --- | --- |
| $$loss_s<0.10s$$ | `tracking` | 完全信任弹道外推，`age_s` 接近真实时间差 |
| $$0.10s\leq loss_s<0.40s$$ | `predicting` | 继续外推，开始让策略感知到观测变旧 |
| $$0.40s\leq loss_s<0.60s$$ | `predicting` | 继续外推，但 `age_s` 向上限增加，让策略降低信任 |
| $$loss_s\geq0.60s$$ | `lost` | 输出 invalid，交给 ball-lost failsafe |

外推时长硬上限：

$$\Delta t_{predict}\leq0.45s$$

如果超过上限但还未达到 invalid 时间，输出会被截断并标记 `clamped=True`。

传给策略的年龄使用分段增长。可信区间内：

$$age_s=loss_s$$

退化区间内：

$$age_s=loss_s+\alpha(age_{ceil}-loss_s)$$

其中：

$$\alpha=\operatorname{clip}\left(\frac{loss_s-0.10}{0.40-0.10},0,1\right)$$

默认 `age_ceiling=0.20s`，与策略的 `rl_ball_obs_age_clip` 对齐。

## 5. 与旧 FOV Loss Compensation 的关系

仓库中仍保留 `fov_loss_compensator.py`。它是一个独立状态机，用在检测器和控制器之间，主要处理球上抛飞出相机视野后的短时预测。

当前推荐使用 `BallStateEstimator`，原因是它同时覆盖：

1. 60Hz 到 200Hz 频率桥接。
2. 感知延迟补偿。
3. FOV 丢失期间的弹道外推。
4. Kalman 融合和异常测量门控。

两者不能同时有效工作。`pingpong_node.py` 中已经做了保护：

| 配置 | 实际行为 |
| --- | --- |
| `enable_ball_state_estimator:=true` 且 `enable_fov_loss_compensation:=true` | 自动关闭 FOV-loss compensator |
| `enable_ball_state_estimator:=false` 且 `enable_fov_loss_compensation:=true` | 使用旧 FOV-loss 状态机 |
| 两者都 false | 控制器直接使用检测器输出 |

## 6. 当前混合检测运行命令

以下命令是当前灰色反光球混合检测和球状态估计的在线运行方式。

```bash
ros2 run pingpong_controller pingpong_node --ros-args \
  -p use_direct_realsense:=true \
  -p enable_rl_policy:=false \
  -p enable_init_pose:=false \
  -p image_width:=848 \
  -p image_height:=480 \
  -p realsense_fps:=60 \
  -p infer_imgsz:=480 \
  -p enable_visualization:=true \
  -p use_kalman:=false \
  -p enable_fov_loss_compensation:=true \
  -p save_raw_video:=true \
  -p raw_video_fps:=60.0 \
  -p enable_ball_state_estimator:=true \
  -p save_estimator_log:=true \
  -p ball_estimator_log_period_s:=0.0 \
  -p use_depth_sensor:=true \
  -p depth_neighborhood_px:=2 \
  -p yolo_model_path:=src/pingpong_moz1/pingpong_controller/models/best.pt \
  -p use_hybrid_detector:=true \
  -p ir_params_json:=ir_params.json \
  -p target_class:='Class 0' \
  -p conf_threshold:=0.25 \
  -p save_ir_frames:=true \
  -p save_depth_frames:=true
```

关键参数解释：

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `use_direct_realsense` | true | 直接由节点或混合检测器管理 RealSense |
| `use_hybrid_detector` | true | 启用 YOLO + IR 混合检测 |
| `target_class` | `Class 0` | YOLO 自定义模型目标类别 |
| `conf_threshold` | 0.25 | YOLO 置信度阈值 |
| `ir_params_json` | `ir_params.json` | IR 阈值、半径、圆度等参数配置 |
| `use_depth_sensor` | true | 启用 RealSense 硬件深度 |
| `depth_neighborhood_px` | 2 | 深度邻域采样半径 |
| `enable_ball_state_estimator` | true | 启用 200Hz 球状态估计 |
| `use_kalman` | false | 关闭传统 RGB 检测器内部 Kalman，不影响 `BallStateEstimator` |
| `save_raw_video` | true | 保存原始 RGB 视频 |
| `save_ir_frames` | true | 保存 IR sidecar，供离线混合检测复现 |
| `save_depth_frames` | true | 保存 Depth sidecar，供离线深度复现 |
| `save_estimator_log` | true | 保存 200Hz 估计器日志 |
| `ball_estimator_log_period_s` | 0.0 | 估计器状态日志以最小节流周期输出，代码最低节流周期为 0.1s |

## 7. 在线输出与日志

### 7.1 ROS 调试话题

`/pingpong/ball_state` 使用 `nav_msgs/Odometry` 发布。

| 字段 | 单位 | 说明 |
| --- | --- | --- |
| `pose.pose.position` | m | 球位置 |
| `twist.twist.linear` | m/s | 球速度 |
| `header.frame_id` | `base_link` | 当前实现固定发布为 `base_link` |
| `child_frame_id` | `pingpong_ball` | 球对象 |

注意：`BallState` 内部单位是 mm 和 mm/s，发布 Odometry 时转换为 m 和 m/s。

### 7.2 原始轨迹文件

真实检测会记录到：

| 文件 | 内容 |
| --- | --- |
| `outputs/pingpong_base_trajectory.csv` | base_link 下真实检测轨迹 |
| `outputs/pingpong_base_trajectory.json` | 同上，JSON 格式 |
| `outputs/pingpong_base_trajectory.png` | 位置和速度曲线 |

预测帧不会写入上述轨迹文件，避免把模型外推结果混入真实检测记录。

### 7.3 估计器 CSV 日志

启用 `save_estimator_log:=true` 后，关闭节点时保存：

```text
src/pingpong_moz1/pingpong_controller/outputs/rl_sim/estimator_log_<timestamp>.csv
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| `t_now` | 200Hz 控制 tick 时间 |
| `valid` | 当前估计状态是否可用 |
| `predicted` | 是否处于超过可信窗口后的预测状态 |
| `state` | `tracking` / `predicting` / `lost` |
| `x_mm,y_mm,z_mm` | 估计器输出位置 |
| `vx_mm_s,vy_mm_s,vz_mm_s` | 估计器输出速度 |
| `age_s` | 策略接收到的观测年龄 |
| `loss_s` | 距离最近真实检测的时间 |
| `extrapolation_s` | 从滤波器时间推进到 `t_now` 的时长 |
| `clamped` | 外推是否被上限截断 |
| `raw_t_img` | 最近真实检测图像时间 |
| `raw_x_mm,raw_y_mm,raw_z_mm` | 最近真实检测位置 |
| `raw_radius_px` | 最近真实检测半径。传统检测通常有值；当前 hybrid raw 字段名为 `radius`，若未同步到 `radius_px`，这里可能为 NaN |
| `raw_depth_mm` | 最近真实检测深度 |

## 8. 离线视频标注方法

混合检测离线标注不是只读取 RGB mp4。它需要同一段录制对应的 sidecar：

| 文件或目录 | 来源 | 作用 |
| --- | --- | --- |
| `<stem>.mp4` | `save_raw_video:=true` | 原始 RGB 视频 |
| `<stem>_ir/ir_<idx>.npy` | `save_ir_frames:=true` | 每帧 IR 图像 |
| `<stem>_depth/depth_<idx>.npy` | `save_depth_frames:=true` | 每帧原始 depth 图 |
| `<stem>_camera_params.json` | `save_ir_frames:=true` 自动写出 | IR/RGB 内参、IR 到 RGB 外参、depth_scale、检测配置 |
| `<stem>_timestamps.csv` | 原始视频录制自动写出 | 每帧真实采集时间 |

离线标注命令：

```bash
python src/pingpong_moz1/pingpong_controller/tools/annotate_raw_video_hybrid.py \
  -i ~/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_2026xxxx_xxxxxx.mp4 \
  --output-frame base_link \
  --output ~/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/videos/pingpong_raw_2026xxxx_xxxxxx_hybrid_annotatedv1.mp4
```

脚本内部复现在线流程：

1. 读取 RGB 视频帧。
2. 按同一 frame index 读取 IR 和 Depth `.npy`。
3. 从 `camera_params.json` 重建 RealSense 内参和 IR 到 RGB 外参。
4. 加载匹配分辨率的手眼标定 JSON。
5. 调用 `get_observation_offline()`，复用在线检测核心。
6. 输出 RGB|IR 双视图标注视频。
7. 输出 `<output>_detections.csv` 检测日志。

离线检测 CSV 字段：

| 字段 | 含义 |
| --- | --- |
| `frame_idx` | 视频帧索引 |
| `t_sec` | 该帧时间 |
| `detected` | 是否检测成功 |
| `x_mm,y_mm,z_mm` | 输出坐标系下位置 |
| `vx_mm_s,vy_mm_s,vz_mm_s` | 速度 |
| `cx,cy` | IR 图像球心 |
| `cx_rgb,cy_rgb` | IR 检测点投影到 RGB 图像的位置 |
| `radius_px` | IR 半径 |
| `depth_mm` | 本帧深度 |
| `depth_method` | `sensor` 或 `geometry` |
| `detection_method` | `hybrid` 或 `ir_fallback` |

## 9. 参数建议与调试入口

### 9.1 检测参数

| 现象 | 优先检查参数或数据 |
| --- | --- |
| YOLO 误检其他反光物 | `target_class`、`conf_threshold`、训练数据 |
| IR 亮斑断裂 | `ir_threshold`、`use_adaptive_ir`、`dilate_iterations`、`close_kernel_size` |
| 球半径异常偏大 | IR 曝光、增益、激光功率、近距离过曝 |
| 深度跳变 | 先看 `depth_method`；近距离 sensor 跳变时不要强行使用硬件深度；geometry 跳变时检查 IR 曝光、半径、形态学膨胀 |
| YOLO 低曝光失效 | RGB 低曝光会让 YOLO 语义检测变差，优先保证 IR 低曝光定位稳定 |
| 在线和离线结果不同 | 检查 `<stem>_camera_params.json` 是否记录了实时检测配置 |

### 9.2 估计器参数

| 现象 | 建议 |
| --- | --- |
| 速度抖动大 | 增大 `ball_estimator_r_depth_mm` 或减小 `ball_estimator_q_depth_vel_mult` |
| 重捕后跟踪慢 | 检查创新门控是否频繁拒绝，必要时查看 `rej/reinit` |
| 短时出 FOV 后过早 invalid | 增大 `ball_estimator_invalid_after_s`，但不要超过弹道模型可信时间 |
| 预测轨迹过冲 | 确认 `ball_estimator_enable_aero_drag:=true`，检查 `drag_k_per_mm` |
| 策略对旧状态过度响应 | 检查 `age_s` 是否按 loss 增长，确认 `rl_ball_obs_age_clip` 与 `degraded_age_ceiling_s` 对齐 |

### 9.3 推荐观察日志

估计器状态日志格式：

```text
BallEstimator: state=tracking valid=True loss=18ms extrap=18ms age=18ms acc=120 rej=0 reinit=0 bounce=0
```

关键判断：

| 字段 | 正常期望 |
| --- | --- |
| `loss` | 检测稳定时通常小于一帧周期到几十 ms |
| `extrap` | 约等于从最近图像时间到控制 tick 的时间差 |
| `age` | 新鲜检测时小，长时间丢失时增长 |
| `rej` | 少量可接受，连续增长说明门控或深度有异常 |
| `reinit` | 击打或反弹后可能增加，静止测试中不应频繁增加 |
| `clamped` | 高抛离开视野很久时可能出现，频繁出现说明外推窗口不足或检测漏失过长 |

## 10. 验收 checklist

| 检查项 | 通过标准 |
| --- | --- |
| 在线检测 | 可视化窗口中 RGB|IR 双视图正常，IR 绿圈稳定锁定球 |
| 深度源 | 记录 `depth_method`，近距离颠球不强求 sensor；关键是深度不出现不可解释的逐帧跳变 |
| 坐标输出 | `Pos[base_link]` 合理，离线 `--output-frame base_link` 能加载标定 |
| 估计器 | 日志中 `BallEstimator` 状态稳定，`valid=True` 时 `/pingpong/ball_state` 连续发布 |
| 原始录制 | mp4、`_ir/`、`_depth/`、`_camera_params.json`、`_timestamps.csv` 同时生成 |
| 离线标注 | 输出 `_hybrid_annotatedv1.mp4` 和 `_detections.csv` |
| 轨迹真实性 | `pingpong_base_trajectory.*` 只包含真实检测，不混入预测帧 |

## 11. 已知边界

1. `BallStateEstimator` 默认不估计自旋，Magnus 项关闭。若球自旋很强，长时间外推仍会有偏差。
2. 感知延迟补偿依赖检测时间戳。当前 hybrid reader 使用 ROS 当前时间作为帧时间；若需要更严格曝光时间，可评估 RealSense hardware timestamp 接入混合路径。
3. 若没有手眼标定或分辨率不匹配，输出会退回 camera 坐标系，`base_link` 轨迹记录和离线标注会受影响。
4. `enable_fov_loss_compensation` 与 `enable_ball_state_estimator` 不应同时作为有效补偿链路使用。当前代码已自动让估计器优先。
5. `use_kalman:=false` 不代表完全没有滤波。混合检测内部仍有 IR 像素级 KF，球状态层还有 `BallStateEstimator`。
6. 近距离高速颠球时，硬件 depth 和单帧几何深度都可能退化。低曝光 IR 中心定位是主观测，深度应结合时域稳定化和状态估计使用。

## 12. 结论

当前方案可以概括为：

1. RGB YOLO 负责“辅助语义约束”，但低曝光下不能作为主检测源。
2. 低曝光 IR 负责“找准像素”，是当前近距离高速颠球最可靠的观测来源。
3. sensor / geometry 深度负责“提供深度候选”，近距离高速场景下需要谨慎使用。
4. `BallStateEstimator` 负责“把 60Hz 真实检测变成 200Hz 当前球状态”。
5. 离线标注脚本通过录制 sidecar 重建在线相机模型，保证在线和离线检测链路尽量一致。

最终控制侧看到的是带有速度、有效性和观测年龄的连续球状态，而不是单帧检测结果。
