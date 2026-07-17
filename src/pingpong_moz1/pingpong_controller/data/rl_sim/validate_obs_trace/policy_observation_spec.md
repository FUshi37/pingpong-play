# MJX PPO Policy Observation Spec

本文档描述 checkpoint `18_stage4g_strong_contact_dr_inverse_mpc.pkl` 的 actor policy 输入观测。当前 actor 输入是 `obs_dim = 67`。checkpoint 里还有 asymmetric critic 的 `critic_obs_dim = 231`，但 validate/部署时 policy 只使用这里列出的 67 维 actor observation。

观测顺序对应 `MjxJuggleEnv._make_obs()` 加 `_delay_conditioning_features()`。

## Checkpoint Context

| Item | Value |
|---|---:|
| Actor observation dimension | 67 |
| Action dimension | 7 |
| Right arm joint order | `RightArm-0`, `RightArm-1`, `RightArm-2`, `RightArm-3`, `RightArm-4`, `RightArm-5`, `RightArm-6` |
| Base observation dimension | 50 |
| Delay-conditioning dimension | 17 |
| High-latency observation expansion | off |
| Delay conditioning | on |
| Delay range | fixed 72 ms |
| Delay jitter | fixed 0 ms |
| Control timestep | 0.005 s, 200 Hz |
| MuJoCo timestep | 0.001 s |
| Ball observation rate | 50 Hz |
| Ball observation age clip | 0.2 s |
| Actuator compensation mode | `inverse_mpc` |

## XML And Coordinate Frames

| Frame / entity | XML/code reference | Definition used by observation |
|---|---|---|
| MuJoCo world frame | `moz1_pd.xml` `<worldbody>` root; floor is `<geom name="floor" pos="0 0 0">`; timestep in `<option timestep="0.001">` | Global inertial MuJoCo coordinates. `+Z` is upward height; gravity acts along negative `Z` in simulation. The XML comment labels the wall at `x=1.2` as robot front, so world `+X` points toward that front-wall side in this scene. |
| Robot floating base body | `moz1_pd.xml` `<body name="base" pos="0 0 0.1">` | Robot root body. The observation does not directly use the full body frame; it uses the planar base joints below. |
| Base planar pose | `moz1_pd.xml` joints `base_x`, `base_y`, `base_yaw` under body `base` | `base_q = [base_x, base_y, base_yaw]`. `base_x/base_y` are world translations; `base_yaw` is yaw about world `Z`. |
| Base yaw frame used in obs | Implemented in `_point_to_base()` / `_vel_to_base()` | A derived planar frame, not a named XML frame. Origin is `(base_x, base_y)` in world XY, orientation is `base_yaw`. `x/y` are translated and yaw-rotated into this base frame; `z` remains MuJoCo world height, not `z - base_z`. |
| Right arm joint frame/order | `moz1_pd.xml` joints `RightArm-0` ... `RightArm-6` | Joint-space quantities follow this order exactly. |
| Racket point | `moz1_pd.xml` `<body name="right_racket">` with `<site name="right_ee_site" pos="0 0 0.007">` | Racket position/velocity observations use the MuJoCo world position of site `right_ee_site`, then transform it into the derived base yaw frame. |
| Racket reset anchor | `reset_racket_anchor = data.site_xpos[:, right_ee_site]` at reset | Ball spawn and episode bounds are measured around this warm/reset racket site position. |
| Ball body | Inserted at runtime by `_build_temp_xml_with_ball()` as `<body name="pingpong_ball">`, `<freejoint name="ball_free">`, `<geom name="ball">` under `<worldbody>` | Ball position/velocity come from the runtime-added MuJoCo ball body/freejoint, then pass through the observation pipeline. |
| Virtual camera | `virtual_camera_body_name = head22`; XML body `<body name="head22">`; mount offset/config in `MjxJuggleConfig` | Used only for validation diagnostics such as `ball_cam_x/y/z` and pixel projection. These camera values are not actor policy inputs. |

## Range Notes

| Category | Meaning |
|---|---|
| Hard range | The value is clipped by XML joint limits or code before entering the observation. |
| Fixed value | For this checkpoint/config the value is constant. |
| Config/termination range | The value is not clipped at observation construction, but the environment reset, controller, or episode termination imposes a practical range. |
| No obs clip | The observation code does not clamp the value; rare transients can exceed the listed practical range. |

Right-arm joint angle limits from `moz1_pd.xml`:

| Joint | Range rad | Range deg | Span rad |
|---|---:|---:|---:|
| `RightArm-0` | `[-2.0944, 3.14159]` | `[-120.0, 180.0]` | `5.23599` |
| `RightArm-1` | `[-2.96706, 0.15708]` | `[-170.0, 9.0]` | `3.12414` |
| `RightArm-2` | `[-3.05433, 3.05433]` | `[-175.0, 175.0]` | `6.10866` |
| `RightArm-3` | `[-0.174533, 2.25147]` | `[-10.0, 129.0]` | `2.42600` |
| `RightArm-4` | `[-3.05433, 3.05433]` | `[-175.0, 175.0]` | `6.10866` |
| `RightArm-5` | `[-1.65806, 1.65806]` | `[-95.0, 95.0]` | `3.31612` |
| `RightArm-6` | `[-1.5708, 1.5708]` | `[-90.0, 90.0]` | `3.14160` |

## Observation Layout

| Obs index range | Name | Dim | Elements | Unit | Coordinate frame | Range / fixed value | Meaning |
|---:|---|---:|---|---|---|---|---|
| 0-6 | `arm_q` | 7 | `RightArm-0..6` | rad | Joint space, XML joints `RightArm-*` | Hard XML joint ranges listed above. | Current right-arm joint angles from MuJoCo `qpos`. |
| 7-13 | `arm_dq` | 7 | `RightArm-0..6` | rad/s | Joint space, XML joints `RightArm-*` | No obs clip. Controller nominal velocity limits are `±[3.66519, 3.66519, 4.18879, 4.18879, 5.23599, 5.23599, 5.23599]` rad/s because `arm_action_limiter=True`; actual MuJoCo joint velocity is observed directly. | Current right-arm joint velocities from MuJoCo `qvel`. |
| 14-16 | `base_q` | 3 | `x`, `y`, `yaw` | m, m, rad | MuJoCo world frame via XML joints `base_x`, `base_y`, `base_yaw` under body `base` | XML joint ranges: `x,y ∈ [-3, 3]` m, `yaw ∈ [-3.14159, 3.14159]` rad. Episode terminates if `|base_x| > 2.6` or `|base_y| > 2.6`. In this validation env, base actuators are commanded to zero. | Mobile-base planar pose used for the base-yaw transform. |
| 17-19 | `base_dq` | 3 | `vx`, `vy`, `yaw_rate` | m/s, m/s, rad/s | MuJoCo world frame velocities of XML joints `base_x`, `base_y`, `base_yaw` | No obs clip. XML velocity actuator ctrl ranges are `Base-X/Y ±1.5 m/s`, `Base-Yaw ±2.5 rad/s`; current MJX policy sets base actuator ctrl to zero, so expected near zero. | Mobile-base planar velocity. |
| 20-22 | `ball_pos_base` | 3 | `x`, `y`, `z` | m | Derived base yaw frame for `x/y`; MuJoCo world height for `z`; ball body `pingpong_ball` | No obs clip. Reset: ball `x/y = right_ee_site x/y ±0.025 m`, `z = right_ee_site z + 0.32 ±0.035 m`. Episode terminates if true ball `z < 0.8` or `z > 1.9`, or `|world x/y - racket_anchor x/y| > 0.5`. Observation noise std is `0.003 m`; obs latency is 0-2 control steps. | Observed ball position after the observation pipeline. |
| 23-25 | `ball_vel_base` | 3 | `vx`, `vy`, `vz` | m/s | Derived base yaw frame for `vx/vy`; MuJoCo world vertical velocity for `vz`; ball body `pingpong_ball` | No obs clip. Reset velocity: `vx,vy ∈ [-0.012, 0.012]` m/s, `vz = -0.28` m/s. Observation velocity noise std is `0.03 m/s`; gravity/contact can change velocity during rollout. | Observed ball velocity after the observation pipeline. |
| 26-28 | `racket_pos_base` | 3 | `x`, `y`, `z` | m | Derived base yaw frame for `x/y`; MuJoCo world height for `z`; XML site `right_ee_site` on body `right_racket` | No obs clip. Episode terminates if `||right_ee_site - racket_anchor|| > 1.1`; also if relative racket height is `> 0.24 m` above anchor or `< -0.12 m` below anchor. | Racket site position. |
| 29-31 | `racket_vel_base` | 3 | `vx`, `vy`, `vz` | m/s | Derived base yaw frame for `vx/vy`; MuJoCo world vertical velocity for `vz`; XML site `right_ee_site` | No obs clip. Computed by finite difference over `dt = 0.005 s`; practical range depends on arm motion. | Racket site velocity. |
| 32-34 | `ball_minus_racket_base` | 3 | `x`, `y`, `z` | m | Derived base yaw frame; difference between ball and `right_ee_site` | No obs clip. At reset, approximately `x/y ∈ [-0.025, 0.025]` m and `z = 0.32 ±0.035 m`. During nonterminal rollout, constrained indirectly by the ball and racket termination bounds above. | Relative position `ball_pos_base - racket_pos_base`; `z` is ball height minus racket height. |
| 35-41 | `prev_action` | 7 | `RightArm-0..6` | normalized, dimensionless | Joint-action order | Hard clipped to `[-1, 1]`; reset value is exactly all zeros. With current config this maps to desired joint acceleration ranges of about `±[19.2859..26.0927, 19.2859..26.0927, 26.7035..36.1283, 44.5059..60.2139, 44.5059..60.2139, 44.5059..60.2139, 44.5059..60.2139]` rad/s² due action-scale domain randomization. | Previous normalized policy action. |
| 42-48 | `arm_cmd_error` | 7 | `RightArm-0..6` | rad | Joint space, XML joints `RightArm-*` | No extra obs clip. Since nominal command target and joint angle are clipped to XML joint ranges, each component is bounded by approximately `±[5.23599, 3.12414, 6.10866, 2.42600, 6.10866, 3.31612, 3.14160]` rad. | Nominal integrated command target angle minus current joint angle: `arm_cmd_q - arm_q`. |
| 49 | `ball_obs_age_norm` | 1 | age | normalized, dimensionless | Scalar | Fixed value `0.0` for this checkpoint because `ball_obs_age_tracks_stale=False`, dropout probabilities are zero, and camera-visible gating is off. General code range is `[0, 1]`. | Ball observation age divided by `ball_obs_age_clip = 0.2 s`. |
| 50 | `tau_act_norm` | 1 | `tau_act / delay_max_s` | normalized, dimensionless | Scalar | Fixed value `1.0` for this checkpoint: `delay_min_ms = delay_max_ms = 72`, `delay_jitter_ms = 0`, `delay_max_s = 0.072`. | Effective actuator/control delay normalized by configured maximum delay. |
| 51-57 | `arm_cmd_qvel` | 7 | `RightArm-0..6` | rad/s | Joint space, XML joints `RightArm-*` | Hard clipped by controller to `±[3.66519, 3.66519, 4.18879, 4.18879, 5.23599, 5.23599, 5.23599]` rad/s. | Current nominal command target velocity for each right-arm joint. |
| 58-64 | `active_command_error` | 7 | `RightArm-0..6` | rad | Joint space, XML joints `RightArm-*` | No extra obs clip. Active reference and joint angle are within XML joint ranges, so each component is bounded by approximately `±[5.23599, 3.12414, 6.10866, 2.42600, 6.10866, 3.31612, 3.14160]` rad. | Currently active delayed joint reference minus current joint angle: `arm_q_ref_active - arm_q`. |
| 65 | `t_contact_est` | 1 | time | s | Scalar | Hard clipped to `[0, 0.5]` s by `max_contact_time = 0.5`. Implementation note: current code uses `base_obs[32]` as the relative-height input, but `base_obs[32]` is `rel_x` in this table. | Environment-estimated time-to-contact feature used by delay conditioning. |
| 66 | `t_margin` | 1 | time | s | Scalar | Because `tau_act = 0.072 s` fixed and `t_contact_est ∈ [0, 0.5]`, range is `[-0.072, 0.428]` s. | Contact-time margin: `t_contact_est - tau_act`. |

## Delay-Conditioning CSV Mapping

In `validate_obs_trace.csv`, raw policy dimensions are stored as `obs/000` to `obs/066`. The last 17 dimensions are also stored as `obs_extra/000` to `obs_extra/016`:

| Policy index | CSV extra alias | Name | Range / fixed value |
|---:|---|---|---|
| 50 | `obs_extra/000` | `tau_act_norm` | fixed `1.0` |
| 51-57 | `obs_extra/001` to `obs_extra/007` | `arm_cmd_qvel` | hard clipped to the joint velocity limits above |
| 58-64 | `obs_extra/008` to `obs_extra/014` | `active_command_error` | bounded by joint spans above |
| 65 | `obs_extra/015` | `t_contact_est` | `[0, 0.5]` s |
| 66 | `obs_extra/016` | `t_margin` | `[-0.072, 0.428]` s |

## Validation-Only Camera Columns

The following columns may appear in `validate_obs_trace.csv` after validation script instrumentation, but they are diagnostics only and are not part of the 67-dimensional actor input:

| CSV columns | Unit | Frame | Meaning |
|---|---|---|---|
| `ball_cam_x`, `ball_cam_y`, `ball_cam_z` | m | Virtual camera optical frame attached to XML body `head22` | True ball position relative to the virtual camera. `ball_cam_z` is optical depth. |
| `ball_cam_distance` | m | Virtual camera optical frame | Euclidean distance from virtual camera optical center to true ball position. |
| `ball_pixel_u`, `ball_pixel_v` | pixel | Camera image plane | Projection using `fx=636.99`, `fy=636.84`, `cx=646.82`, `cy=373.21`, image size `1280x720`. |
| `camera_visible`, `camera_in_front`, `camera_in_depth`, `camera_in_frustum`, `camera_in_image` | boolean as 0/1 | Camera visibility logic | Visibility diagnostics. Camera-depth range is `0.15 m` to `2.50 m`. |

## Source Code References

| Component | Source |
|---|---|
| Base observation construction | `pingpong_controller/tools/rl_sim/mjx_juggle_env.py`, `_make_obs()` |
| Base-yaw transform | `pingpong_controller/tools/rl_sim/mjx_juggle_env.py`, `_point_to_base()` and `_vel_to_base()` |
| Delay-conditioning features | `pingpong_controller/tools/rl_sim/mjx_juggle_env.py`, `_delay_conditioning_features()` |
| Contact-time estimate | `pingpong_controller/tools/rl_sim/mjx_juggle_env.py`, `_estimate_contact_time_from_obs()` |
| Observation CSV aliases | `pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py`, `obs_row_from_host()` |
| XML model | `pingpong_controller/tools/rl_sim/moz1_pd.xml` |
