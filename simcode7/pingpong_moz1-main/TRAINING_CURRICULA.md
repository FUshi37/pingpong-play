# Training Curriculum Registry

Last reconciled: 2026-08-22. This registry records released and actively referenced courses; executable truth remains the profile builder plus the exact launcher snapshot. Update this file in the same commit as any profile, preset, observation, action, DR, gate, checkpoint, or control-stack change.

## Registry Requirements

Every new course entry must record:

- immutable profile name/version and status (`released`, `active`, `paused`, `experimental`, or `retired`);
- action semantics and integration feedback source;
- actor/critic observation dimensions and append/migration contract;
- plant/planner path, XML, PD profile, control/observation rates, and DR preset/ranges;
- ordered stage names or unambiguous stage groups, gates, and zero-based resume index;
- parent checkpoint, SHA-256, seed, PPO settings, launcher, GPU/run/W&B identity;
- validation commands, preregistered metrics, result paths, and the decision supported by the evidence.

Never edit a released profile in place. Add a new monotonic version and keep the old builder loadable for checkpoint replay.

## Released Baselines

| ID | Status | Version identity | Course | Published checkpoint |
| --- | --- | --- | --- | --- |
| GPU0-QVEL | released | artifact `e012d07400a99e95b50c2dcf7200d6ede312dfa6`; companion code parent `d269517c4e79cc53b9024dc6bdf9eff496a653d3` | `goal_d455_sport_taskspace_qvel_vertical_v14` | `pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/mjx_curriculum_last.pkl` |
| GPU1-QACC | released | model `4f47e6d3c67917acd6a9b05162a488c91b319597`; companion stack child `69b9bf3442f5e9473bb291e0fa9439d0f15a53d5` | `goal_d455_sport_taskspace_obsres2mm_nocomp_direct_v1` | `pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl` |

### GPU0-QVEL: v14 Heavy-Ball Baseline

Contract:

- Actor/action: `67 -> 256 -> 256 -> 7`, `action_command_mode=velocity`, velocity scale `0.85`; actor output passes a 15 ms normalized-action LPF, slew limit `30/s`, the physical qvel/qacc envelope, and one integration into a q-only target.
- Critic: asymmetric 231-D training-only input (67 actor + 80 legacy privileged + 84 command-history dimensions).
- Plant: delayed per-joint second-order actuator, `sport_taskspace_fit_v1` simulation PD, no compensation and no servo planner. Actuator DR is frequency/damping scale `[0.90,1.10]`, gain `[0.99,1.01]`, delay offset `[-2,+1]` at 5 ms.
- Timing: 200 Hz policy/control and 60 Hz fractional ball observations. Ball observation position/velocity noise is 0.002 m / 0.07 m/s; observation latency DR is 0--2 control steps.
- Released checkpoint SHA-256: `2ca715d71fe19a0058b8cac710574630bdff580ec28157ac80a3e1d70f8cef47`.
- Final continuation: GPU0, seed 61010, 1024 envs, 128 rollout steps, minibatch 16384, 4 epochs, LR `2e-5`, gamma `0.9995`, GAE `0.99`, clip `0.12`, target KL `0.008`, log-std `[-3.2,-2.5]`, entropy `0.0006`, anchor coefficient `0.015`.
- Canonical launcher and evidence: `pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812/launch_formal_gpu0_v14_online.sh` and `pingpong_controller/tools/rl_2real/GPU0_QVEL_V14_HEAVY_BALL_REAL_ROBOT_DEPLOYMENT.md`.

The 24 zero-based stages are:

```text
00 qvel_v14_quality_lock_after_rebound
01 qvel_v14_angular_velocity_polish
02 qvel_v14_low_motion_full_orbit_proof
03 qvel_v14_racket_launch_first_hit_bridge
04 qvel_v14_launch03_ball_dynamics_mild
05 qvel_v14_launch04_contact_dynamics_mild
06 qvel_v14_launch05_actuator_pd_mild
07 qvel_v14_launch06_racket_geometry_mild
08 qvel_v14_launch07_observation_calibration_mild
09 qvel_v14_launch08_single_dropout_preview
10 qvel_v14_launch09_camera_missing_mild
11 qvel_v14_launch10_workspace_wide
12 qvel_v14_launch11_ball_dynamics_wide
13 qvel_v14_launch12_contact_dynamics_wide
14 qvel_v14_launch13_actuator_pd_wide
15 qvel_v14_launch14_racket_geometry_wide
16 qvel_v14_launch15_observation_calibration_micro_bridge
17 qvel_v14_launch16_observation_calibration_three_eighths_bridge
18 qvel_v14_launch17a_refresh_noise_only_long_juggle_1200
19 qvel_v14_launch17b_frame_dr_only_long_juggle_1200
20 qvel_v14_launch17c_measured_obsres2mm_sport_nocomp_long_juggle_1200
21 qvel_v14_launch19_final_measured_obsres2mm_sport_nocomp_consolidation
22 qvel_v14_heavy_ball_mass_elasticity_bridge
23 qvel_v14_heavy_ball_3p7g_lower_elasticity_target
```

Stages 0--3 acquire a low-motion recurrent orbit; 4--10 add mild ball/contact/actuator/geometry/observation/dropout axes; 11--17 widen them and bridge measured observation error; 18--21 prove the 1200-step horizon; 22 covers 2.90--3.70 g with solref damping 0.66--1.06; 23 targets 3.45--3.95 g with damping 0.72--1.08. The last gate is 13 mean hits and 0.95 mean length fraction; release promotion additionally required 128-env full-length/view validation.

### GPU1-QACC: Pure-Actuator v5 Baseline

Contract:

- Actor/action: 67-D actor observation, 7-D joint-acceleration action; command-state `qdd -> dq_cmd -> q_cmd` double integration.
- Plant: per-joint delay, underdamped second-order actuator, then `sport_taskspace_fit_v1` position PD. Compensation, inverse MPC, teacher, servo planner, and state governor are disabled.
- Actuator DR: frequency/damping scale `[0.90,1.10]`, gain `[0.99,1.01]`, delay offset `[-2,+1]` control steps.
- Published checkpoint: actor 67-D, action 7-D, stage 21, global step 1,533,116,416; SHA-256 `9d7e94e9ef803fcbe9385ab97485626b2529394be62c75136c69d873adffaa79`.
- Recovery launcher: GPU1, seed 976, 640 envs, 128 steps, minibatch 16384, 4 epochs, LR `2e-4`, clip `0.15`; from zero-based stage 18 use 256 steps, 2 epochs, LR `5e-5`. See `pingpong_controller/tools/rl_sim/launch_gpu1_pure_stable_v5_recovery.sh`.
- Canonical model contract: `pingpong_controller/tools/rl_2real/GPU1_PURE_ACTUATOR_V5_REAL_ROBOT_DEPLOYMENT.md`.

The 21 zero-based stages are:

```text
00 launch00_acquisition
01 launch01_local_workspace
02 launch02_workspace
03 launch03_ball_dynamics_mild
04 launch04_contact_dynamics_mild
05 launch05_actuator_pd_mild
06 launch06_racket_geometry_mild
07 launch07_observation_calibration_mild
08 launch08_single_dropout_preview
09 launch09_camera_missing_mild
10 launch10_workspace_wide
11 launch11_ball_dynamics_wide
12 launch12_contact_dynamics_wide
13 launch13_actuator_pd_wide
14 launch14_racket_geometry_wide
15 launch15_observation_calibration_micro_bridge
16 launch16_observation_calibration_three_eighths_bridge
17 launch17a_refresh_noise_only_long_juggle_1200
18 launch17b_frame_dr_only_long_juggle_1200
19 launch17c_measured_obsres2mm_sport_nocomp_long_juggle_1200
20 launch19_final_measured_obsres2mm_sport_nocomp_consolidation
```

The gates progress from 1.0 hit / 0.10 length at stage 0 through 11.8 / 0.90 at stage 14, use relaxed observation bridges at 15--16, require 12.0 / 0.90 in the long-horizon stages, and finish at 13.0 / 0.95. Ball-mass DR is 2.60--2.80 g in mild stages and 2.45--2.95 g in the wide/final stages.

## Sim-to-Real Fine-Tuning Courses

All courses in this section preserve the released command-state integration semantics. They are not evidence for changing an existing checkpoint to actual-`q/dq` feedback.

### GPU0 QVEL v38 Full-Horizon

Status: paused/abandoned for current scheduling, retained for audit and reproducibility.

- Intended profile: `goal_d455_sport_taskspace_qvel_sim2real_full_horizon_v38`, 40 stages (0--39), action QVEL, `q_cmd/dq_cmd` integration.
- Launch snapshot: `pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/resume_gpu0_v38_privcritic_20260818.sh`.
- Formal settings: GPU0, seed 82705, 1024 envs, 128 steps, minibatch 16384, 2 epochs, LR `3e-5`, clip `0.08`, target KL `0.003`, 48-update convergence window, asymmetric critic plus 12-step command history and 48-D sim-to-real block (67-D actor / 279-D critic).
- Course groups: 0--2 source-height bridge; 3--6 plant-center 25/50/75/100%; 7--10 jointwise residual common 0.5 and 0.0 plus consolidation; 11--18 measured residual 25/50/75/100% plus consolidation; 19--26 execution hold 3%, 8%, tail-half, and tail-full plus consolidation; 27--36 impact-half/full, latency-full, 75/90 Hz observation plus consolidation; 37 B2 low-energy recovery; 38 B3 high-energy normal-impulse recovery; 39 combined proof.
- Normal stage gates are approximately 11.5 hits / 0.90 length with at least 80 updates; consolidation uses 12.0 / 0.92 with at least 128; final proof uses 12.5 / 0.95 with at least 220 plus strict hit/view/truncation gates.

Known discrepancy: the dated handoff and launcher describe a 40-stage QVEL course, but the current `build_curriculum()` direct dispatch for the v38 profile does not invoke `_sport_taskspace_qvel_sim2real_full_horizon_v38_stages()` and presently returns the 21-stage **acceleration-mode** base course. Do not resume or reuse v38 until a targeted contract test demonstrates both QVEL action semantics and the intended ordered 40-stage output. Fixing this is a separate code task and must not mutate the old checkpoint semantics.

### GPU1 QACC v120, v122, and v123 Hit-Aligned Continuation

Status: v120 is the fine-tune base. The v122 GPU1 job was stopped at residual-100 update 1112 on 2026-08-21 after its 48-update hit window plateaued below the 11.5-hit gate. V123 is the opt-in successor; it preserves the actuator/checkpoint lineage while changing only the late reward and the explicitly documented PPO credit horizon. The reviewed V123 formal run started in `pp_gpu1` at 2026-08-21 19:01 CST, using output directory `formal_gpu1_v123_hit_aligned_long_credit_20260821` and W&B id `v123g1a1`.

Shared contract:

- QACC with `q_cmd/dq_cmd` command-state integration; 67-D actor and 279-D asymmetric critic (`--critic-command-history-steps 12 --critic-sim2real-privileged`).
- Recalibrated record_new3 second-order actuator DR; no switch to recovered RMP within this checkpoint lineage.
- 44 ordered stages (0--43): 0--1 source height; 2--5 plant center; 6--10 jointwise residual/retention/consolidation; 11--14 ball-mass bridge and 3.7 g consolidation; 15--22 measured residual 25/50/75/100% and consolidations; 23--30 execution holds; 31--40 impact/latency/observation; 41--42 B2/B3 recovery; 43 combined proof.
- Adaptation gates generally require 11.5 hits / 0.90 length / at least 80 updates; consolidation requires 12.0 / 0.92 / at least 128; final proof requires 12.5 / 0.95 / at least 220, with strict conditional-hit and truncation gates.

The exact 44 stage names are:

```text
00 record_new3_sim2real_v119_source_height_1_0p22
01 record_new3_sim2real_v119_source_height_2_0p23
02 record_new3_sim2real_v119_plant_center_25
03 record_new3_sim2real_v119_plant_center_50
04 record_new3_sim2real_v119_plant_center_75
05 record_new3_sim2real_v119_plant_center_100
06 record_new3_sim2real_v119_jointwise_residual_common_0p5
07 record_new3_sim2real_v119_jointwise_residual_common_0p5_retention_recovery
08 record_new3_sim2real_v119_jointwise_residual_common_0p5_consolidate
09 record_new3_sim2real_v119_jointwise_residual_common_0p0
10 record_new3_sim2real_v119_jointwise_residual_common_0p0_consolidate
11 record_new3_sim2real_v119_ball_mass_bridge
12 record_new3_sim2real_v119_ball_mass_bridge_consolidate
13 record_new3_sim2real_v119_ball_mass_3p7g
14 record_new3_sim2real_v119_ball_mass_3p7g_consolidate
15 record_new3_sim2real_v119_measured_residual_25
16 record_new3_sim2real_v119_measured_residual_25_consolidate
17 record_new3_sim2real_v119_measured_residual_50
18 record_new3_sim2real_v119_measured_residual_50_consolidate
19 record_new3_sim2real_v119_measured_residual_75
20 record_new3_sim2real_v119_measured_residual_75_consolidate
21 record_new3_sim2real_v119_measured_residual_100
22 record_new3_sim2real_v119_measured_residual_100_consolidate
23 record_new3_sim2real_v119_hold_3pct_single
24 record_new3_sim2real_v119_hold_3pct_single_consolidate
25 record_new3_sim2real_v119_hold_8pct_single
26 record_new3_sim2real_v119_hold_8pct_single_consolidate
27 record_new3_sim2real_v119_hold_8pct_tail_half
28 record_new3_sim2real_v119_hold_8pct_tail_half_consolidate
29 record_new3_sim2real_v119_hold_8pct_tail_full
30 record_new3_sim2real_v119_hold_8pct_tail_full_consolidate
31 record_new3_sim2real_v119_impact_half_60hz
32 record_new3_sim2real_v119_impact_half_60hz_consolidate
33 record_new3_sim2real_v119_impact_full_60hz
34 record_new3_sim2real_v119_impact_full_60hz_consolidate
35 record_new3_sim2real_v119_latency_full_60hz
36 record_new3_sim2real_v119_latency_full_60hz_consolidate
37 record_new3_sim2real_v119_observation_75p0hz
38 record_new3_sim2real_v119_observation_75p0hz_consolidate
39 record_new3_sim2real_v119_observation_90p0hz
40 record_new3_sim2real_v119_observation_90p0hz_consolidate
41 record_new3_sim2real_v119_b2_low_energy_residual_recovery
42 record_new3_sim2real_v119_b3_high_energy_normal_impulse_recovery
43 record_new3_sim2real_v119_main_racket_launch_combined_proof
```

V120 profile: `goal_d455_sport_taskspace_record_new3_sim2real_full_horizon_v120`. The stage-15 resume snapshot uses seed 82731, 1024 envs, 128 steps, minibatch 16384, 2 epochs, LR `3e-5`, clip `0.08`, target KL `0.003`, entropy `1e-4`, log-std `[-3.8,-3.4]`, and anchor coefficient `0.01`. See `pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/resume_gpu1_v120_privcritic_stage15_20260819.sh`.

V122 profile: `goal_d455_sport_taskspace_record_new3_sim2real_delay_prior_v122`. It keeps the v120 stage order and every measured delay-support endpoint, but from stage 17 (`...measured_residual_50`) onward samples the nominal zero-offset delay component with probability `0.75`. The latest launcher resumes at zero-based stage 18 with seed 82731 and adds time-limit bootstrap, residual L2 `0.01`, the same PPO scale, W&B id `c6va1bpt`, and a frozen stage-17 actor anchor:

`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v122_delay_prior.sh`

The script selects `CUDA_VISIBLE_DEVICES=1`, while its default run directory, W&B name, and tags contain `gpu0`. This label/device mismatch must be explicitly resolved or overridden before any new launch. The launcher is also untracked at this reconciliation point, so preserve a reviewed script snapshot and run manifest before treating a new run as reproducible. Do not rename or interrupt an already running formal job merely for cosmetic consistency.

V123 profile: `goal_d455_sport_taskspace_record_new3_sim2real_hit_aligned_v123`. Stages 0--20 are exactly V122. From zero-based stage 21 (`...measured_residual_100`) onward, the stage names use the `record_new3_sim2real_v123_` prefix and only these reward fields change: `hit_count_floor_reward_weight 5.0 -> 8.0`, `hit_apex_view_center_penalty_weight 1.0 -> 0.0`, and `ball_view_xy_center_penalty_weight 0.75 -> 0.375`. The counted-hit cap remains 14, so the quality-independent floor is bounded to 112 reward per episode. Every apex, camera, view, cadence, recovery, truncation and safety graduation gate remains unchanged.

The reward change was selected before V123 training by replaying the V122 residual-100 telemetry from updates 16--1110. The original 48-update maximum-hit window ended at update 403 with 11.271 hits, but its mean per-step reward was 0.001713 below the terminal update-1110 window with 11.022 hits. The V123 counterfactual makes the higher-hit window better by 0.000510 reward/step while retaining half of the dense view-center pressure. This is an observational counterfactual, not training proof; V123 must still satisfy the unchanged gates.

The frozen V123 source and actor anchor are both the V122 residual-100 update-400 checkpoint, SHA-256 `ec53c07ba5175eb68a2ab0d79e83feb295d3b623adf35b4d60ce1946f2ff0212`. A deterministic, paired 64-environment GPU1 screen used the update-400 environment and seed 82731. The historical stage-17 checkpoint is actually tagged `...measured_residual_25_consolidate` and scored 10.16 hits / 0.688 full episodes / 0.931 hit lower-band rate. Update 400 scored 11.36 / 0.844 / 0.964; the stopped update-1110 checkpoint scored 11.31 / 0.859 / 0.968. Update 400 was selected on the preregistered primary hit metric because its view metrics pass and its large V122 rolling window also had the highest hits.

V123 changes the launcher rollout from 128 to 256 control steps and GAE lambda from 0.99 to 0.995; gamma remains 0.9995. At 200 Hz this extends direct on-policy context from 0.64 s to 1.28 s. The exponential GAE trace time constant rises from about 0.48 s to 0.91 s, so an action retains about 0.32 rather than 0.11 trace weight across two measured 0.52 s hit intervals. Every other PPO, DR, gate, seed and checkpoint-regularization setting remains V122-identical.

The reviewed launcher is:

`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v123_hit_aligned_long_credit.sh`

It selects physical GPU1 UUID `GPU-e74458ec-002a-1bac-e0be-d0a5713b661e`, starts one-based stage 22, and uses a new run directory and W&B identity because the reward/GAE semantics changed. W&B remains online, metrics-only, in project `pingpong-mjx`; reusing V122 id `c6va1bpt` is forbidden because it would merge incompatible histories.

V123 was safely stopped on 2026-08-24 after update 12314 in zero-based Stage
22, `record_new3_sim2real_v123_measured_residual_100_consolidate`. Its only
complete rolling-gate pass during the long plateau held for one update rather
than the required 16; terminal windows remained near the unchanged 12-hit and
0.92-length gates without an improving trend. The fixed diagnostic source is
the Stage-22 update-8400 archive, SHA-256
`592f1a1586e08aa0181f10284c42b977e5876bede078c33bcb3d107a0b027319`.
The frozen update-400 actor anchor remains SHA-256
`ec53c07ba5175eb68a2ab0d79e83feb295d3b623adf35b4d60ce1946f2ff0212`.

Profiles V124--V128 are opt-in GPU1 plateau diagnostics. V124--V127 preserve
all 44 V123 course slots, QACC command-state action semantics, 67-D actor/279-D
critic, actuator/observation/DR physics and every graduation gate. From
zero-based Stage 22 onward only the following versioned reward fields differ:

- V124 `goal_d455_sport_taskspace_record_new3_sim2real_gate_aligned_v124`
  reduces anti-correlated local-Y/view shaping and doubles bounded apex credit.
  Its 96-update trial was negative: best complete windows reached only about
  12.011 hits / 0.910 length.
- V125 `goal_d455_sport_taskspace_record_new3_sim2real_terminal_aligned_v125`
  returns to V123 shaping and sets the three fixed early-termination barriers
  to 20.0. Its trial was negative at about 11.952 hits / 0.913 length.
- V126 `goal_d455_sport_taskspace_record_new3_sim2real_duration_aligned_v126`
  retains V125 and adds `post_first_hit_alive_reward_weight=3.0`. Its complete
  trial ended near 11.924 hits / 0.91 length; full/view/apex/cadence and safety
  remained healthy, but no complete strict-gate window passed.
- V127 `goal_d455_sport_taskspace_record_new3_sim2real_duration_commit_v127`
  changes only that direct alive weight from 3.0 to 5.0. The maximum ordinary
  six-second contribution remains below 30 reward, versus V123's bounded 112
  hit-floor reward. Its trial was negative: the terminal complete window had
  12.097 hits / 0.910 length and 0.820 full rate. PPO and safety remained
  healthy, so direct alive-weight scanning is closed.

V128 `goal_d455_sport_taskspace_record_new3_sim2real_temporal_actuator_v128`
returns exactly to V123 rewards, DR, gates and PPO settings. Starting only at
zero-based Stage 22 it enables the proven four-frame causal observation/action
layout, migrating the 67-D actor to 254-D and the 279-D asymmetric critic to
466-D. The old actor rows and 17 delay-conditioning rows retain their exact
weights, new temporal rows start at zero, and the existing 212 critic-only
rows are shifted behind the expanded actor block. This exposes measured
response history but never realized frequency/damping/gain/delay DR values to
the actor. The fixed update-8400, seed-82731 bounded trial was negative. At
update 96 its complete 48-update window had 12.089 hits, 0.9057 episode-length
fraction, 0.8161 full-episode rate, 0.9373 true-view fraction and 0.2238 m
relative apex. The hit gate and all listed secondary gates passed, but the
unchanged 0.92 length gate never passed (`hold_count=0`). Mean last-12 exact
KL was 0.00218, qvel exceedance was 0.000064 and qacc exceedance was 0.00292,
so the trial was optimizer- and safety-stable but did not resolve the survival
plateau. Do not promote or run V128 without an update cap. V128 is not
deployable through the released 67-D GPU1 adapter; a matching 254-D causal
history runtime requires separate tests before any robot use.

V129 `goal_d455_sport_taskspace_record_new3_sim2real_geometric_survival_v129`
is the user-authorized survival-reward successor. It returns to the V123
67-D actor/279-D critic, optimizer, physics, DR, action semantics, rewards and
gates, and from zero-based Stage 22 changes exactly one scalar:
`post_hit_survival_reward_weight 2.0 -> 3.0`. This term is active only after a
counted hit while the ball remains above the racket anchor and scores centered,
bounded-horizontal-velocity recovery. The failed V126/V127 unqualified
post-first-hit alive term remains zero, and V123 low-ball/termination/hit/apex/
view rewards remain exact. The additive `+1.0` follows the GPU0 isolated
survival-credit ablation scale without copying GPU0's control stack or its
rejected lower-apex weakening. Formal eligibility requires the fixed
update-8400, seed-82731 production-shape trial to pass every original complete
48-update Stage-23 gate for the 16-update hold with protected PPO and qvel/qacc
safety. The reviewed launcher is
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v129_geometric_survival.sh`;
it restarts the immutable V123 update-8400 optimizer lineage in a new output
directory and new W&B run, with no stage cap and block validation, only after
that bounded acceptance passes.

The preregistration, all negative trials and checkpoint-screen evidence are
recorded in
`pingpong_controller/outputs/rl_sim/gpu1_v123_plateau_experiments_20260824/V124_PLATEAU_EXPERIMENT_REPORT.md`.

## Evidence Behind the Fine-Tune Courses

The primary record_new3--5 evidence bundle is under `pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/`:

- `RECORD_NEW3_5_REAL_FAILURE_ROOT_CAUSE_REPORT.md`: end-to-end causal chain and evidence boundary.
- `FAILURE_MODE_REPRODUCIBILITY_ADDENDUM.md`: repeatable B2/B3 failure branches and intervention logic.
- `ACTUATOR_DR_COVERAGE_AND_RACKET_KINEMATICS_20260814.md`: joint/racket coverage analysis and why continuous DR was not widened further.
- `SIMULATION_SIM2REAL_MODIFICATION_CODEX_PROMPT.md`: simulation changes, staged experiments, and preregistered acceptance gates.
- `REAL_ROBOT_PLATFORM_MODIFICATION_CODEX_PROMPT.md`: real-platform observation/plant/logging contract and staged hardware gates.

Use the real implementation at `/home/yangzhe/Project/pingpong_playreal/pingpong-play` only as a cross-repository reference unless explicitly authorized to edit it.

## Experimental From-Scratch QVEL Curriculum: Measured Feedback + RMP/PD

Status: `experimental`, executable profile implemented, formal launch blocked on
new-ball measurements and RMP/PD/ball-outcome DR coverage. It has no parent or
checkpoint and must not be described as active or released.

- Immutable profile: `goal_d455_measured_qvel_rmp_vertical_v1`.
- Launcher: `pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v1.sh`.
- XML: `pingpong_controller/tools/rl_sim/moz1_pd.xml`, adapted at load time by
  `right_arm_pd_profile=recovered_rmp_equiv_v1`.
- Actor/action: 50-D measured observation and 7-D QVEL. Dimensions 0--41 retain
  measured kinematics, ball/racket geometry, relative position, and previous
  policy action; dimensions 42--48 are encoder `dq(t)-dq(t-1)` and dimension
  49 is ball-observation age. There are no `q_cmd/dq_cmd` fields. Each policy
  step computes its q target from current simulated/encoder `q/dq`.
- Why not the released 67-D layout: its extra 17 values are actuator tau,
  `dq_cmd`, active command error, and two actuator/contact phase values. They
  are control-stack-specific and invalid for the no-actuator RMP profile.
  Input dimension is not network capacity (the actor hidden width remains
  256); adding unneeded or privileged inputs can instead create a sim-to-real
  shortcut. A future measured-history augmentation requires a new version and
  an ablation rather than silently changing v1.
- Critic: 214-D asymmetric simulator-only input with 12 RMP command-history entries;
  no actor observation migration or actuator-checkpoint resume is allowed.
- Rates: 200 Hz policy, five recovered-RMP and MJX ticks per policy step at
  1 kHz. The RMP output delay is `[18,19,19,18,18,18,18]` ms.
- Control exclusivity: no actuator command filter/model, FOPDT, inverse
  compensation, second planner, plant command delay, or actual-state
  projection. All prior actuator profiles retain their original path.

The executable simulation chain is:

```text
policy integration target
  -> 5 x 1 ms recovered joint RMP
  -> q/qd/qdd references at each substep
  -> XML low-level PD
  -> 5 x 1 ms MJX physics
```

The ordered zero-based stages are:

```text
00 rmp_launch00_acquisition
01 rmp_launch01_local_workspace
02 rmp_launch02_workspace
03 rmp_launch03_ball_dynamics_mild
04 rmp_launch04_contact_dynamics_mild
05 rmp05_rmp_pd_mild
06 rmp_launch06_racket_geometry_mild
07 rmp_launch07_observation_calibration_mild
08 rmp_launch08_single_dropout_preview
09 rmp_launch09_camera_missing_mild
10 rmp_launch10_workspace_wide
11 rmp_launch11_ball_dynamics_wide
12 rmp_launch12_contact_dynamics_wide
13 rmp13_rmp_pd_coverage
14 rmp_launch14_racket_geometry_wide
15 rmp_launch15_observation_calibration_micro_bridge
16 rmp_launch16_observation_calibration_bridge
17 rmp_launch17_observation_calibration_wide
18 rmp_launch18_camera_missing_wide
19 rmp_launch19_final_consolidation
```

Stages 0--2 use nominal RMP/PD and the measured mass midpoint; 3--4 add mild
ball/contact physics; 5 adds mild RMP/PD; 6--10 independently add geometry,
observation, missingness, and workspace; 11--12 use the full measured mass,
inertia `[0.40,2/3]`, reset spin x/y `[-55,55]` rad/s and z `[-40,40]`
rad/s, and full contact support; 13 adds the candidate coverage envelope;
14--19 widen observation/task axes and consolidate.

The immutable final candidate RMP/PD DR is: RMP Kp/Kd `[0.75,1.25]`, estimator
process `[0.60,1.60]`, estimator measure `[0.50,2.00]`, velocity feed-forward
`[0.35,0.65]`, acceleration weight `[0.50,2.00]`, target window `[8,12]`,
output-delay offset `[-3,+3]` ms, PD Kp/Kv `[0.70,1.30]`, damping
`[0.65,1.45]`, and armature `[0.65,1.50]`. These remain candidate values until
the exact range manifest passes; a changed range requires a new profile
version.

The course reuses the proven 20-stage launch/workspace/physics/observation
ordering and the successful GPU0-QVEL physical-hit semantics instead of
inventing an unrelated task: the 0.22 s anti-chatter debounce and explicit
fast-recontact penalty are retained. It does not inherit GPU0-QVEL V14's low
0.18 m orbit. All stages target a 0.23 m post-hit apex; stages 10--19 must
graduate inside the fine-tune `0.218--0.248 m` mean apex band and a physical
cadence window with minimum interval `0.36 s` (the inherited later upper gates
remain no looser than `0.58 s`).

Behavior shaping targets the D455 view/contact center, allows only bounded
inward local-Y recovery up to 0.10 m/s, penalizes approach/contact/cycle racket
horizontal motion and full angular speed, and preserves racket flatness. The
last gate requires mean/rms contact racket horizontal velocity at most
`0.07/0.12 m/s`, contact angular speed at most `0.8 rad/s`, and racket-up
cosine at least `0.96`, in addition to the inherited hit/episode/view gates.

Formal GPU0 defaults are seed `20260820`, 512 envs, 128 rollout steps,
minibatch 8192, 4 epochs, LR `3e-4`, gamma `0.9995`, GAE `0.99`, clip `0.2`,
target KL `0.012`, hidden width 256, strict curriculum gates, blocking
next-stage validation, and W&B metrics-only upload. The launcher checks the
physical GPU0 UUID, current processes, git status, XML/evidence hashes, and a
fresh run/W&B identity before it accepts `CONFIRM_GPU0_READY=YES`.

Evidence status:

- Training-local RMP versus 3590-point DataTracer output:
  `0.038381 deg / 0.010817 rad/s / 0.303445 rad/s2`; passes the documented
  `0.1 deg / 0.05 rad/s / 1 rad/s2` RMP-layer gates.
- Delayed final reference versus encoder on that trace:
  `0.156158 deg / 0.078217 rad/s / 10.200634 rad/s2`; fails and remains a
  low-level servo/mechanical/feedback residual, not an RMP recovery error.
- Full training path on known divergent record_new4/203833:
  `5.691621 deg / 0.499332 rad/s / 8.215055 rad/s2`; retained as a negative
  result consistent with `ReactiveMotionPlanner/rmp-recovery/RMP.md`.
- Corrected GPU0 diagnostic coverage on the first 100 samples of
  record_new3/193409 (0.15 s warmup, 8 independent plus 3 stress candidates)
  failed every component family. Worst pointwise margins included
  `-0.0452 rad` joint position, `-0.4702 rad/s` joint velocity,
  `-19.08 rad/s2` joint acceleration, and `-0.0158 m` racket position.
  This is an explicit negative result, not enough records/candidates for the
  formal gate, and proves that the v1 numerical ranges remain hypotheses.
- Report and artifacts:
  `pingpong_controller/outputs/rl_sim/rmp_measured_qvel_design_20260820/`.

The launcher still requires a validated manifest generated from record_new3,
record_new4, and record_new5 with at least 64 usable recordings, 64 independent
DR candidates, per-sample/per-axis joint/racket position/orientation/velocity/
acceleration observed-support coverage from hashed NPZ artifacts,
worst-component plots, and at least 30
ball-outcome trials. It also requires at least three replacement-ball mass and
diameter measurements and coverage of free flight, restitution, tangential
response, spin, and apex gain. The schema and validator are
`rmp_dr_coverage_manifest_v1.template.json` and `rmp_training_evidence.py`.
Generate the RMP/PD half of that evidence with
`quantify_measured_qvel_rmp_dr_coverage.py`; it uses fixed-size candidate waves
to replay the exact training-local RMP, output delay, adapted XML PD, and MJX
arm path without compiling unrelated reward/contact code. It intentionally
leaves ball outcome evidence incomplete.

No such passing manifest or replacement-ball measurements exist in this
workspace as of this reconciliation, so no GPU training has been started.
After the evidence gate passes, run QVEL from scratch and require a fixed-seed
GPU smoke plus multi-seed convergence and frozen-checkpoint validation before
promotion. QACC remains a separate future course and must not resume this or an
actuator checkpoint.

## Experimental Measured-QVEL + RMP/PD V2 Correction

Status (2026-08-21): `experimental`, implemented, and intentionally blocked;
no PPO training was started. Immutable profile
`goal_d455_measured_qvel_rmp_vertical_v2` supersedes v1 for future evidence,
but v1 remains available for reproduction. The GPU0 launcher is
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v2.sh`.

V2 retains the requested 50-D actor, 214-D asymmetric critic, measured `q/dq`
QVEL integration, 0.23 m apex objective, `0.218--0.248 m` late apex gate,
center/vertical/racket-motion shaping, and the 20-stage ordering. It remains
from-scratch and cannot load either a 67-D actuator checkpoint or the v1
course. The 50-D choice is semantic, not a capacity reduction: the removed
17 values describe actuator tau, `dq_cmd`, active command error, and legacy
phase; they do not exist in the RMP stack. Network capacity remains the
256-wide hidden representation. Adding measured history would be a separate
observation ablation/profile, not padding to 67 dimensions.

### Corrected control and timing contract

The canonical JSON contains global and per-joint XML-PD scales, but the batch
command behind RMP.md uses the per-joint values as replacements. V1 multiplied
both layers and therefore used Kp/Kv that were 50x/10x too small. V2 selects
`right_arm_pd_profile=recovered_rmp_rmpmd_v2` with effective gains:

- Kp: `[112320,184320,133650,73200,30420,62100,35100]`.
- Kv: `[9048,9604.8,9570,4071.6,3712,5000,2030]`.

On record_new3/193409, 300 samples of the V2 training-local path match the
canonical RMP.md batch path to `5.1e-7 rad` q RMS and `1.33e-4 rad/s` dq RMS
(maximum differences `4.2e-6 rad` and `0.00168 rad/s`). Both produce joint
tracking RMS `0.011421 rad / 0.20114 rad/s / 4.80383 rad/s2` on that window.
The independent 3590-point RMP-output gate remains
`0.038381 deg / 0.010817 rad/s / 0.303445 rad/s2`, passing RMP.md thresholds.

Real controller stalls are represented before RMP, not as a plant model:

```text
50-D actor + measured q/dq QVEL integration
  -> optional 200 Hz target-publication ZOH
  -> five 1 ms recovered-RMP updates
  -> recovered RMP output delay
  -> RMP.md-effective XML PD
  -> five 1 ms MJX steps
```

Stages 0--7 use no target holds, 8--12 use start probability `0.025`, 13--15
use `0.05`, and 16--19 use `0.08`. The late tail probability is `0.022` with
2--9 held ticks. `dr_execution_command_hold_probability` remains zero and the
environment rejects enabling both hold locations. These values come from the
record_new3/4/5 wall-clock publication audit; they are scheduler timing, not
FOPDT, an actuator filter, or a second planner.

### Candidate V2 DR and current negative evidence

The late-stage candidate envelope ramps from the v1 mild range at stages
13--15 and reaches the following stable bounds at stages 16--19:

| factor | final range |
| --- | --- |
| RMP joint Kp/Kd multiplier | `[0.50,1.50]` / `[0.50,1.50]` |
| estimator process/measure multiplier | `[0.25,2.50]` / `[0.01,3.00]` |
| velocity feed-forward | `[0.20,0.80]` |
| acceleration-weight multiplier | `[0.25,3.00]` |
| target-filter length | `[5,15]` 1 kHz ticks |
| RMP-output delay offset | `[-8,+5]` 1 kHz ticks |
| XML-PD Kp/Kv multiplier | `[0.50,1.50]` / `[0.50,1.50]` |
| dof damping/armature multiplier | `[0.40,1.80]` / `[0.40,2.00]` |

These are candidate ranges only. The preregistered wall-clock coverage contract
uses a uniform 5 ms `wallTimeS` grid, zero-order-held published targets, and
joint-header-time interpolation. `sourceTimeS` is retained only for paired
diagnosis because it compresses missed controller ticks. Formal v2 evidence
also hashes `rmp_pd_candidate_stability.npz` and recomputes finite-state,
joint-limit, velocity-utilization (`<=1.10`), and acceleration-utilization
(`<=2.00`) gates; a divergent candidate cannot manufacture a wide envelope.

GPU0 worst-four-record screens used 300 samples, 64 independent LHS candidates
plus 41 structured candidates, the corrected PD, and no PPO training:

| screen | q | dq | qdd | racket families | stable |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 ranges + wall time | 65.0% | 62.5% | 50.2% | 71.3--85.5% | not recorded |
| bounded V2 multifactor | 90.1% | 92.6% | 84.9% | 96.2--99.1% | 105/105 |
| wider timing ablation | 90.1% | 93.0% | 85.5% | 95.9--99.1% | 105/105 |

The timing ablation did not improve worst margins consistently and was
rejected. The bounded V2 envelope is safer and narrower, but it still fails
the required pointwise real-tail gate; there is no formal 64-record/full-length
pass. Artifacts are under
`pingpong_controller/outputs/rl_sim/rmp_measured_qvel_design_20260820/`, in
`coverage_gpu0_worst4_300sample_wall_rmpmd_bounded_multifactor_v2/` and the
paired diagnostic directories.

### Effective-juggling population and minimum-range correction

The next course must not interpret all parseable rows as normal robot
behavior. The pre-DR contract is
`pingpong_controller/tools/rl_sim/rmp_effective_juggling_validity_v1.json`.
It is hashed into each diagnostic and defines validity independently of the
candidate envelope:

- record: longest contiguous published-policy segment >=300 rows, no safety
  intervention, and all three recording groups retained;
- sample: usable ball state with age <=50 ms, backed by either `rawValid` or
  `predictedBallIsPredicted`, ball center in `[0.90,1.75] m`, at least 5 mm
  above the racket center, within 0.30 m horizontally, and speed <=6 m/s;
- nominal-replay anomaly: reject only when both racket position RMS and angle
  RMS exceed the fixed 2026-08-20 `Q3 + 3*IQR` outer fences (`0.022793 m` and
  `2.68656 deg`) in the hashed 64-record report.

The 2026-08-21 full read-only audit found 65 directories, parsed 64, and
accepted 56 records: 15 new3, 19 new4, and 22 new5. Six exclusions were too
short, two were joint position+angle replay outliers, one of the short records
also contained contact-action-shield intervention, and one additional
directory had fewer than two usable rows. The accepted records contain 27,165
wall-grid rows; 24,402 (89.83%) pass the effective-sample rules. Every accepted post-warmup sample
still requires 100% pointwise joint/racket component support. Range selection
must minimize each interval subject to that coverage and candidate stability;
percentile coverage or post-hoc deletion is not sufficient.

The validity correction removed a no-evidence stale-ball segment from
`record_new5/152353` (`rawValid=false`, no active predictor) while retaining
the legitimate predicted high-ball phase in `record_new5/161403`. On these
two effective probes, an 8-LHS+41-stress screen improved the racket families
to 97.7--100% but still left joint q/dq/qdd worst fractions
90.6%/92.0%/88.3%. Two isolated hypotheses were rejected:

- increasing only the RMP output-delay offset upper bound from +5 to +80 1-kHz
  ticks did not improve the joint gap;
- directly applying the copied Movax YAML joint-impedance torque limits
  `[45,35,25,25,10,10,10] Nm` made q/dq/qdd coverage collapse to
  39.3%/15.9%/12.5%, so those numbers are not a valid direct MuJoCo actuator
  mapping for this joint mode.

Therefore the current blocker is nominal RMP/low-level closed-loop
calibration or a missing physical factor, not permission to expand every DR
range. V2 remains immutable and blocked; the effective-population result must
feed a new monotonically versioned profile only after a narrow, stable, fully
covering envelope exists. The aborted global-padding attempt under
`rmp_measured_qvel_rmp_v2_full_wall_20260821/` is explicitly marked invalid.
Future audits use 256-step length buckets to avoid padding every trace to the
longest recording.

The immutable V2 launcher therefore requires all of the following before it can reach the
GPU confirmation step:

1. A V2 manifest from all 64 usable record_new3/4/5 recordings, at least 64
   independent candidates, full-duration wall alignment, pointwise hashed
   joint/racket artifacts, worst-component curves, and 100% declared support.
2. Every sampled candidate passing the hashed stability artifact.
3. At least three replacement-ball mass and diameter measurements, with the
   training mass interval containing all measurements.
4. At least 30 physical outcome trials covering free flight, restitution,
   tangential response, spin, and apex gain.

Generate the arm half from `pingpong_controller/tools/rl_sim/` with:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MPLCONFIGDIR=/tmp/pingpong_rmp_coverage_matplotlib \
/home/yangzhe/miniconda3/envs/pingpong/bin/python \
  quantify_measured_qvel_rmp_dr_coverage.py \
  --profile goal_d455_measured_qvel_rmp_vertical_v2 \
  --right-arm-pd-profile recovered_rmp_rmpmd_v2 \
  --time-alignment wall \
  --control-dt-s 0.005 \
  --lhs-samples 64 \
  --recording-wave-size 2 \
  --candidate-wave-size 8 \
  --rmp-pd-ranges-json \
    ../../outputs/rl_sim/rmp_measured_qvel_design_20260820/rmp_pd_range_search_bounded_multifactor_v2.json \
  --output-dir \
    ../../outputs/rl_sim/rmp_measured_qvel_rmp_v2_full_wall_20260821
```

Do not add `--allow-incomplete-coverage` to formal evidence. The command
deliberately leaves ball evidence incomplete; use
`rmp_dr_coverage_manifest_v2.template.json` for the combined schema. Once a
genuinely passing manifest exists,
run the launcher first without `CONFIRM_GPU0_READY`; inspect git/GPU/hash/W&B
identity, then set `CONFIRM_GPU0_READY=YES`. QACC remains a later independent
from-scratch course.

### Deployment boundary

The recovered RMP used above is the simulator-side surrogate for the RMP that
already runs on the physical robot. It is not deployed. The real chain is
encoder `q/dq` -> 50-D actor -> QVEL -> measured-feedback-integrated 200 Hz
position target -> `MechUnitCmd.jnt_pos` -> robot RMP/low-level controller.
Do not run recovered RMP, XML PD, its simulated output delay, or target-hold DR
again in the policy process. Real missed publications naturally hold the last
robot-RMP target.

This model boundary is directly deployable on a robot whose existing RMP
accepts that seven-angle position target. Real-runtime source changes are not
part of this training task; the eventual runtime only needs to reproduce the
fail-closed measured-feedback 50-D observation/action/target contract. The
exact formula, metadata checks, logging fields and rollout order are in
`pingpong_controller/tools/rl_2real/MEASURED_QVEL_RMP_V2_DEPLOYMENT_CONTRACT.md`.

## Experimental Measured-QVEL + RMP/PD V3 Manual Launch

Status (2026-08-21): superseded after the first GPU0 run. The run at
`measured_qvel_rmp_vertical_v3_gpu0_seed20260821_20260821` reached update 184
with zero confirmed hits. Its final updates still contained roughly 900
physical ball/racket contact edges per rollout, but zero launch-clearance
crossings and zero confirmed hits; the process was stopped and its artifacts
were preserved. Do not restart V3 from scratch or resume its checkpoints.

Immutable profile `goal_d455_measured_qvel_rmp_vertical_v3` freezes V2 exactly:
50-D measured-feedback actor, 214-D asymmetric critic, 7-D QVEL action,
integration from current `q/dq`, recovered RMP, RMP.md-effective XML PD, the
20-stage center/vertical/apex course, target-publication holds, and the bounded
V2 RMP/PD DR table above. V3 has no parent checkpoint and remains from-scratch.

The only V3 difference is evidence disposition. The user accepts the current
bounded DR for an initial experiment and defers detailed RMP tuning. V3 does
not claim complete encoder-feedback coverage: after the two fixed nominal
outliers were removed, two hard valid probes still had joint q/dq/qdd
pointwise coverage of `90.62% / 91.99% / 88.30%`; the RMP output layer itself
still passes `0.038381 deg / 0.010817 rad/s / 0.303445 rad/s2`. The isolated
low-PD-bandwidth probe did not materially improve joint coverage. V1/V2 retain
their strict manifest gates.

Launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v3.sh`.
Use kilograms and supply bounds from the ball that will actually be used.
First run preflight only:

```bash
NEW_BALL_MASS_MIN_KG=<measured_min_kg> \
NEW_BALL_MASS_MAX_KG=<measured_max_kg> \
bash pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v3.sh
```

After reviewing git status, GPU0 UUID/processes, hashes, run directory and W&B
identity, start manually:

```bash
NEW_BALL_MASS_MIN_KG=<measured_min_kg> \
NEW_BALL_MASS_MAX_KG=<measured_max_kg> \
CONFIRM_GPU0_READY=YES \
ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE=YES \
bash pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v3.sh
```

The bypass is profile-restricted and explicit. It does not allow a resume,
command-state integration, an actuator model/filter, inverse compensation,
post-RMP plant hold, a second planner, or actual-state projection. The real
deployment boundary remains measured encoder `q/dq` -> actor QVEL -> integrated
seven-angle target -> robot-resident RMP.

## Experimental Measured-QVEL + RMP/PD V4 Hit Discovery

Status (2026-08-21): superseded after the short GPU0 integration run. Immutable profile
`goal_d455_measured_qvel_rmp_vertical_v4` preserves the complete V3
measured-feedback QVEL, RMP/PD, observation, task, and bounded DR contract.
It fixes only the demonstrated from-scratch acquisition failure by prepending
one centered `falling_contact` stage before the unchanged V3 `racket_launch`
course. The integration run produced 577 physical contact edges per rollout
but still zero launch-clearance crossings and zero confirmed hits: realized
racket vertical speed remained near zero. The run was stopped without resume;
its directory is
`measured_qvel_rmp_vertical_v4_gpu0_seed20260821_20260821_hitfix`.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v4.sh`.
Its default ball-mass DR is `[0.0025, 0.0040] kg` (2.5--4.0 g), seed is
`20260821`, and it remains from-scratch only. The V3 experimental evidence
bypass remains explicit, while all no-stacking and GPU identity checks remain
enforced.

## Experimental Measured-QVEL + RMP/PD V5 Guided First Launch

Status (2026-08-21): stopped and rejected. Immutable profile
`goal_d455_measured_qvel_rmp_vertical_v5` fixes the isolated V4 acquisition
failure without changing hit counting. Its centered falling-contact discovery
stage exposes the full existing bounded QVEL target range and activates the
existing successful-cycle task-space `racket_z/racket_vz` teacher before hit
one. The teacher is reward-only: it supplies neither observations nor actions,
and is not part of deployment. The remainder of the 21-stage course preserves
V4/V3 control, physics, DR, safety, and task settings.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v5.sh`.
It starts from scratch on GPU0, defaults ball-mass DR to
`[0.0025, 0.0040] kg` (2.5--4.0 g), uses measured 50-D feedback and 7-D QVEL,
and retains the exclusive recovered-RMP -> RMP.md XML-PD path. The phase
reference is used only for training reward credit and cannot alter the direct
deployment boundary of integrated angle targets sent to the robot RMP.

The GPU0 run was stopped at update 5565 after approximately 364.7 million
environment steps. It never produced a confirmed hit: the final window had
about 495 physical contact edges per rollout, zero launch-clearance crossings,
RMP `qd` norm about `0.173 rad/s`, and only about 2.9% measured joint-velocity
utilization despite policy action norm about 2.13. Do not resume V5.

## Experimental Measured-QVEL + RMP/PD V6 No Teacher

Status (2026-08-21): from-scratch GPU0 training candidate. A deterministic
unit test and CPU MJX/RMP smoke test pass for the corrected input contract.
Immutable profile `goal_d455_measured_qvel_rmp_vertical_v6` preserves V5's
physical course and full first-stage normalized QVEL range while removing all
phase-teacher references, strengths, and q/dq/racket-z/racket-vz reward
weights from every stage. The command line rejects attaching a phase-teacher
reference to V6.

V6 changes only the RMP input contract: policy QVEL is velocity-clipped, then
integrated as `q_target = clip(q_measured + qvel_policy * 0.005 s)` and passed
to recovered RMP. No policy-side `ddq` is calculated or clipped on this path;
the recovered RMP is the sole source of executed `q/qd/qdd`. V1--V5 retain the
old acceleration-limited integration behavior for reproducibility.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v6.sh`.
It starts from scratch on physical GPU0, defaults ball-mass DR to
`[0.0025, 0.0040] kg` (2.5--4.0 g), and has no teacher input or reward.

Formal launch (2026-08-21): tmux session `pp_gpu0`, run directory
`measured_qvel_rmp_vertical_v6_gpu0_seed20260821_20260821_directqvel_offline`,
offline W&B id `6pdktfdl`. JAX reported `CudaDevice(id=0)` on physical GPU0
`GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`. The preceding online-W&B attempt
`..._directqvel` was interrupted before JAX initialization after repeated API
timeouts; it contains no training evidence and must not be resumed.

V6 was stopped at update 202 (13,238,272 environment steps) after the first
stage remained at zero displayed hits. The interrupted checkpoint is retained
only as negative evidence (SHA-256
`10895a05b132e11de476fbb9ac5268d10a884cbb806a53fb0ea9f44af8f73b47`).
The run recorded 105,772 physical contact edges but only 87 launch-clearance
crossings and two confirmed events over 104,847 completed episodes. Its
one-tick measured-q target was therefore a control-authority defect, not a
missing-contact or counter-only failure. See
`V6_ZERO_HIT_BUG_DIAGNOSIS_20260821.md` in the run directory. Do not resume V6.

## Experimental Measured-QVEL + RMP/PD V7 Full Scale + 17.5 ms Lead

Status (2026-08-21): opt-in, from-scratch GPU0 candidate pending PPO smoke.
Immutable profile `goal_d455_measured_qvel_rmp_vertical_v7` preserves every V6
observation, reward, reset, curriculum gate, RMP/PD, physics and DR setting.
It retains no phase teacher and makes exactly two control-authority changes:
all stages use `action_velocity_scale=1.0`, and direct measured-QVEL targets use

```text
q_target = clip(q_measured + qvel_policy * 0.0175 s)
```

V1--V6 remain unchanged; a zero configured lead remains the legacy one-control-
period behavior. The 17.5 ms value was selected on physical GPU0 from the
preregistered `12.5/15/17.5 ms` screen. It produced confirmed hits for 4/7
stroke onsets at each of 2.5, 3.7 and 4.0 g, versus 2/7 for 15 ms and 0/7 for
12.5 ms at 3.7 g. Peak actual qvel/qacc utilization was 0.278/0.869 and both
exceedance fractions were zero. This is simulator evidence only. Full rows and
the decision report are under
`measured_qvel_rmp_lead_screen_scale1_20260821_mass_edges_v1/`.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v7.sh`.
It uses `--n-envs 1024`, `--n-steps 128`, `--minibatch-size 16384`, and four
PPO epochs, starts from scratch, defaults to offline W&B, and retains the V3--
V6 explicit incomplete-RMP-DR acknowledgement. The one-update 1024-env smoke
had 1,134 contact edges, 185 clearance crossings, six effective low launches,
and zero confirmed hits; qvel/qacc exceedance was 0/0.00000327, within the
unchanged 0.5%/1.0% course gates. An early-monitored formal run is permitted
by the passing oracle screen, but must be stopped if confirmed hits do not
leave zero rather than repeating the long V6 zero-hit run.

Formal launch (2026-08-21): tmux session `pp_gpu0`, run directory
`measured_qvel_rmp_vertical_v7_gpu0_seed20260821_20260821_lead17p5ms_scale1`,
offline W&B id `38bkd97j`. JAX reported `CudaDevice(id=0)` on physical GPU0.
By update 20 (2,621,440 environment steps), per-rollout confirmed events had
risen from zero to 31, mean hits to 0.0259, and effective launches to 120 while
subfloor launches fell from a recent 248 to 189. Qvel exceedance remained zero
and qacc exceedance was at most 0.0000153, far below its 0.01 gate. This early
trend clears the V6 zero-hit stop condition but is not curriculum convergence;
continue monitoring the normal hit, survival, view, and safety gates.

The user stopped V7 for a command-feedback ablation at update 82 and global
step 37,748,736. The final displayed window had mean hits about 0.995 and 896
confirmed events; qvel exceedance remained zero, while qacc exceedance was
about 0.01054 and therefore slightly above the 0.01 course gate. The retained
interrupted checkpoint SHA-256 is
`e187cb7bc6f5f0057c9e015bedab2c3291bfd321e2f07b4dfdf3b04c281e848f`.
Do not silently resume it as a different feedback contract.

## Experimental RMP QVEL Command-Feedback Ablation V8/V9

Status (2026-08-21): explicit diagnostic profiles. V8 is the paired measured-q
control: it preserves V6's one normal 5 ms integration step, sets every stage's
`action_velocity_scale=1.0`, and adds no target lead. V9 differs from V8 only
by using the prior internal `q_cmd` as the next target-integration anchor:

```text
q_target(t) = clip(q_cmd(t-1) + qvel_policy(t) * 0.005 s)
```

The actor observation remains the same 50-D encoder-only measured-feedback
layout. V9 does not expose `q_cmd` to the actor and does not add an actuator
model, qacc integrator, phase teacher, compensation, post-RMP hold, or second
planner. The command-feedback behavior requires an explicit RMP-only ablation
flag; all normal measured-RMP profiles continue to reject command feedback.

Canonical V9 launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v9.sh`.
It starts from scratch with `--n-envs 1024`, `--n-steps 128`,
`--minibatch-size 16384`, and four PPO epochs. Treat V9 only as a trend/root-
cause experiment: if hits improve, that identifies measured-q re-anchoring as
the authority loss but does not establish command-state deployment fidelity.

Formal launch (2026-08-21): tmux session `pp_gpu0`, run directory
`measured_qvel_rmp_vertical_v9_gpu0_seed20260821_20260821_qcmd5ms_scale1`,
offline W&B id `132or6q8`. JAX reported `CudaDevice(id=0)` on physical GPU0.
The random/early actor immediately produced nonzero hits (updates 1--5 mean
hits 0.064, 0.141, 0.178, 0.191, 0.155), showing that command accumulation
restores physical stroke authority without extra lead. By update 19 it had
fallen to about 0.01, so learnability/convergence is not yet established.
Across the first eight recorded updates qvel exceedance was zero and qacc
exceedance was 0.00488--0.00693, below the 0.01 course gate.

By update 224 (29,360,128 environment steps), V9 had collapsed to displayed
mean hits 0.00 and mean episode length fraction about 0.03. GPU0 was already
idle when the next ablation was requested. Do not resume V9 as evidence that
command-state integration fixes the zero-hit failure.

## Experimental Measured-QVEL Fitted-Actuator Stage-1 Ablation V10

Status (2026-08-21): from-scratch, first-stage-only root-cause experiment.
Profile `goal_d455_measured_qvel_actuator_stage1_ablation_v10` preserves V8's
discovery reset, reward, gates, action scale and encoder-only 50-D actor
observation. It changes only the control stack. Policy QVEL is integrated
directly from measured q for one 5 ms control period:

```text
q_target = clip(q_measured + qvel_policy * 0.005 s)
```

Neither measured/command dq nor a policy-side ddq calculation or acceleration
limit participates. The target then traverses the released per-joint delayed
second-order sport actuator and `sport_taskspace_fit_v1` XML-PD profile instead
of recovered RMP. There is no phase teacher, actuator compensation, servo
planner, second planner, or command state in the actor observation. Nominal
plant parameters are used so actuator DR cannot obscure this first isolation.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_actuator_stage1_ablation_v10.sh`.
It uses 1024 environments, 128 rollout steps, minibatch size 16384, four PPO
epochs and `--max-stages 1`. This diagnostic is not a released deployment
contract and must start from scratch.

Formal launch (2026-08-21): tmux session `pp_gpu0`, run directory
`measured_qvel_actuator_stage1_ablation_v10_gpu0_seed20260821_20260821_qmeas_direct5ms`,
offline W&B id `h8fycv9w`. JAX reported `CudaDevice(id=0)` on physical GPU0;
the environment audit reported `obs_dim=50`, `delay_extra_dim=0`, fitted
actuator filtering enabled, and every compensation/planner block disabled.
Updates 1--5 had zero confirmed hits despite 915--1,277 physical contact edges
per rollout; launch-clearance crossings were only 0--2. Qvel/qacc exceedance
fractions were both zero. This was an early-trend snapshot; the run was later
stopped as recorded below.

V10 was stopped for the final command-anchor ablation at update 83 and global
step 10,878,976. Mean hits remained exactly zero. The safely interrupted
checkpoint SHA-256 is
`100056f5e97f8b2564b3ac39b0e8ce715605a3d7e3421d0cad34a48c6e78e317`.

## Experimental Command-QVEL Fitted-Actuator Stage-1 Ablation V11

Status (2026-08-21): final paired root-cause experiment. Profile
`goal_d455_command_qvel_actuator_stage1_ablation_v11` is identical to V10
except for its hidden integration anchor:

```text
q_target(t) = clip(q_cmd(t-1) + qvel_policy(t) * 0.005 s)
```

It still does not use measured/command dq, calculate policy ddq, or apply a
policy-side acceleration limit. The fitted delayed second-order actuator,
sport XML-PD, first-stage reset/reward/gates, PPO settings and 50-D encoder-
only actor observation remain unchanged; q_cmd is not exposed to the actor.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_command_qvel_actuator_stage1_ablation_v11.sh`.
It starts from scratch with 1024 environments, 128 rollout steps, minibatch
size 16384, four PPO epochs and `--max-stages 1`.

Formal launch (2026-08-21): tmux session `pp_gpu0`, run directory
`command_qvel_actuator_stage1_ablation_v11_gpu0_seed20260821_20260821_qcmd_direct5ms`,
offline W&B id `v07b3ku4`. JAX reported `CudaDevice(id=0)` on physical GPU0;
construction reported `obs_dim=50`, `delay_extra_dim=0`, fitted actuator
filtering enabled and no compensation/planner. Unlike V10, V11 immediately
produced nonzero hits: updates 1--5 mean hits were 0.115, 0.231, 0.233, 0.256
and 0.178, with 522--650 clearance crossings in updates 1--5 after the first.
Qvel exceedance remained zero and qacc exceedance was 0.00169--0.00209.
Mean hits then declined to about 0.01 by update 17. Thus q_cmd accumulation
restores initial physical stroke authority, but the shared PPO/task contract
is still learning away that behavior.

Final audit: the GPU0 process received SIGINT after update 195 and stopped
safely after completing update 196 / global step 25,690,112. Mean hits reached
exactly zero by update 64 and remained zero through update 196; over the same
interval mean episode length fell to 24.02 simulation steps while mean return
improved from the early-hit trough of about -3.17 to -2.556. The final policy
therefore did not merely lose contact authority: it learned a higher-return
early-termination strategy. Physical GPU0 is idle after the stop; the
independent GPU1 process was not touched. Final artifact hashes are:

- `mjx_curriculum_interrupted.pkl` and `mjx_curriculum_last.pkl`:
  `cba458d6093e81dd1df75a1817cd8328c4fba5a0ba440ae68404d08030b2af4f`
- `mjx_curriculum_best.pkl`:
  `9655aabffb9f6af861a3a9bd8c1aefe7e86fa3438a03f526517f1abd03f02fa2`
- `curriculum_progress.csv`:
  `7d88884338c1f5daa4846dfc6aab07d1dc28e2f1b939d3e68173d31d4970ca74`

### Shared first-stage reward/termination defect

The 2x2 control ablations are confounded by a shared curriculum-objective
defect and must not be used to conclude that measured-q integration or RMP is
the sole cause. At V11 update 196, `done/racket_too_low=0.041603` per step and
`reward/racket_z_limit_termination_penalty=-0.104008` per step. Their ratio is
exactly the configured fixed 2.5 terminal cost. No ball-miss termination or
confirmed hit remained. The reciprocal termination rate also predicts the
observed approximately 24.0-step episode length. PPO had learned to drive the
racket below its hard workspace bound immediately and pay one cheap terminal
cost. The final rollout had KL 0.00520, explained variance 0.318, zero qvel
exceedance and qacc exceedance fraction 0.1467; the latter also fails the
course's 0.01 safety gate but is downstream of the already-conclusive reward
hack.

The source is the profile construction, not evidence of a GAE regression.
Every measured-QVEL RMP profile and both actuator ablations are built from the
old `goal_d455_autolaunch_v1` minimal-reward course. That source fixes
`post_hit_survival_reward_weight=0`, `center_flat_hit_reward_weight=0`, and
ball/racket terminal bases at 2.5. The measured-QVEL adapter then enables
first-stage apex/view/height, racket-vxy, angular-speed and stability penalties
without restoring the released acquisition rewards or stronger terminal
barriers. In V11 updates 1--4, the selected event-local positive terms were
only about 1.18--1.19 per confirmed hit while the selected event-local
penalties were about 2.08--2.34, making a confirmed hit net negative even
before subsequent trajectory cost. By contrast, the released GPU1-QACC
launch00 configuration uses post-hit survival 1.4, center-flat hit reward 2.4,
and miss/racket terminal bases 3.3/3.5; the released GPU0-QVEL course retains
post-hit survival 1.4, a positive center-flat hit term, and still stronger
failure barriers.

The current GAE implementation is unchanged across the released GPU0-QVEL
source boundary for this path, and all 11 focused GAE regression tests pass.
V12 was pre-registered to repair only the acquisition objective first while
preserving the 50-D observation, measured-q integration and recovered RMP
stack. Its completed negative run below disproved the hypothesis that reward
repair alone was sufficient: it removed the deliberate early-exit policy but
could not produce one confirmed event through the one-tick target. The paired
V13 diagnostics and bounded PPO run now show that additional target authority
is required by the current position-only RMP interface.

## Experimental Measured-QVEL RMP Reward Repair V12

Status (2026-08-21): implemented and regression-tested; no PPO run has been
started by this repository change. Profile
`goal_d455_measured_qvel_rmp_vertical_v12` inherits V8's complete 21-stage
course and changes only reward accounting. It preserves the measured-q
integration path, recovered RMP with `recovered_rmp_rmpmd_v2`, 50-D
encoder-only actor observation, no phase teacher, `action_velocity_scale=1`
in every stage, and `recovered_rmp_qvel_target_lead_s=0`. The zero configured
extra lead retains the ordinary one-control-tick target:

```text
q_target = clip(q_measured + qvel_policy * 0.005 s)
```

The repair retains the existing vertical/contact quality gradients while
making the task objective monotonic in valid contacts. Every counted hit gets
a quality-independent 1.5 reward floor; center/flat hit reward is restored to
2.4; post-hit survival reward is restored to 1.4; and the hit-count cap is
disabled. For a stage with target `H=max(1,target_mean_hits)`, terminal costs
are fixed before training as:

```text
miss_base = 2.5 + 0.8 * H
racket_z_base = racket_anchor_base = 2.5 + 1.0 * H
all corresponding per-earned-hit terminal coefficients = 0
```

This follows the released GPU1-QACC acquisition scale and the GPU0-QVEL
count-credit/net-positive repair principle. It prevents a later failure from
clawing back already-earned hit reward and removes the 2.5-cost deliberate
`racket_too_low` exit found in V11, without deleting height, view, racket-vxy,
angular-speed, stability, cadence, or anti-contact-cheat terms.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v12.sh`.
It is from-scratch only and fixes PPO at 1024 environments, 128 rollout steps,
minibatch size 16384 and four epochs. W&B mode is unconditionally `online` in
the launcher. The normal GPU0 UUID check, unique output directory, RMP-output
replay validation and explicit incomplete-DR acknowledgement remain required.
The launcher performs preflight only unless `CONFIRM_GPU0_READY=YES` and
`ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE=YES` are both supplied.

Pre-registered early acceptance: confirmed hits must leave zero without
`racket_too_low` becoming the dominant early-termination mode, while the
unchanged qvel/qacc course safety gates continue to pass. A higher return with
zero hits or shortened episodes is an explicit failure, not convergence.

Formal outcome (2026-08-21): stopped and rejected. The first online W&B
attempt `..._reward_repair_lead0_scale1` timed out before JAX initialization
and contains no training evidence. The retry ran on physical GPU0 with JAX
`CudaDevice(id=0)`, online W&B id `pxclnejg`, and directory
`measured_qvel_rmp_vertical_v12_gpu0_seed20260821_20260821_reward_repair_lead0_scale1_retry1`.
It received `SIGTERM` after update 56, completed update 57 / 7,471,104 steps,
saved `mjx_curriculum_last.pkl`, and exited cleanly. Across 61,482 episodes it
recorded 63,235 physical contact edges and 56 launch-clearance crossings, but
zero confirmed hit events in every update. `done/racket_too_low` and both
qvel/qacc exceedance fractions remained zero; maximum actual qvel/qacc
utilization was only 4.27%/8.86%. The reward repair therefore removed the
known cheap-exit strategy but did not repair the one-tick measured-q control-
authority defect. Artifact SHA-256 values are:

- `mjx_curriculum_last.pkl`: `ee31031c789d42d53ef9201673f32a4ece1c6ee79c971824689517f3af594896`
- `mjx_curriculum_best.pkl`: `efeb80607cb622c8a98827a4f1d14e90fd91653bc2d0657e96e3665aa02dc8bf`
- `curriculum_progress.csv`: `2ed6f07ff56d92f3681868fc97f06958cdb6f33b3acb8d804d44f5a8003bfc8f`
- `stdout_stderr.log`: `a00174ada1cbee4d82e471883f786060476646d57b2ad7503e3dc04524910ec3`

Do not resume V12.

## Experimental Measured-QVEL RMP Reward Repair + Lead V13

Status (2026-08-21): opt-in, from-scratch single-variable root-cause
experiment. Profile `goal_d455_measured_qvel_rmp_vertical_v13` inherits all
21 V12 stages and changes only
`recovered_rmp_qvel_target_lead_s: 0 -> 0.0175`. Its target is:

```text
q_target = clip(q_measured + qvel_policy * 0.0175 s)
```

The actor remains a 50-D encoder-only measured-feedback network with 7-D
QVEL action, `action_velocity_scale=1`, no command-state feedback and no phase
teacher. Reward, resets, gates, PPO settings, recovered RMP, RMP.md PD, XML,
DR and no-stack checks are byte-for-byte V12 except for names/notes and this
target horizon. The 17.5 ms value is not a new sweep: it is the smallest
pre-registered V7 candidate that produced confirmed hits for 4/7 stroke
onsets at each of 2.5, 3.7 and 4.0 g with no qvel/qacc exceedance.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v13.sh`.
It preserves V12's 1024 environments, 128 rollout steps, minibatch 16384,
four PPO epochs, LR `3e-4`, gamma `0.9995`, GAE `0.99`, clip `0.2`, target KL
`0.012`, entropy `0.0002`, online W&B, from-scratch/no-resume rule and explicit
incomplete-RMP-DR acknowledgement. Formal use remains online by default;
`LOCAL_OFFLINE_SMOKE=YES` is an explicit local-only diagnostic override and
does not change the formal default.

Pre-registered bounded-smoke acceptance: at least one of updates 1--5 must
have nonzero mean hits; by update 20 mean hits should exceed `0.01`; qvel and
qacc exceedance must remain within `0.005`/`0.01`; and `racket_too_low` must
not become a dominant exit. Passing proves that target lead restores early
PPO acquisition under the repaired objective; it is not convergence or formal
hardware-coverage evidence.

Root-cause diagnostic outcome (2026-08-21): the new deterministic GPU0
diagnostic used seven fixed 3.7 g falling-ball lanes and the same maximal
warm-pose vertical-Jacobian action in every case. The V12 effective 5 ms target
produced 0/7 confirmed-hit lanes, maximum racket vertical speed 0.576 m/s, and
maximum qvel/qacc utilization 8.77%/26.85%. Mean policy-QVEL-to-RMP-estimator,
RMP-output and measured-velocity norm gains were only 7.23%/5.63%/5.40%.
Changing only the target horizon to 17.5 ms produced 4/7 confirmed-hit lanes,
1.927 m/s racket vertical speed, 27.80%/86.89% qvel/qacc utilization, no limit
exceedance, and gains of 23.96%/17.45%/15.77%. On identical 17.5 ms dynamics,
the old V8 reward assigned successful lanes mean return -3.008 while the V12
repair assigned -0.080. Thus V12's reward repair is directionally correct,
but its counted-hit credit is unreachable under the one-tick control map.
Artifact:
`measured_qvel_rmp_path_reward_diagnosis_20260821/diagnosis.json`, SHA-256
`dd4726a5eecf5c742cec3261d8e06b1875498d754f537a26d230490843105f36`.

Bounded PPO outcome (2026-08-21): pass for minimum acquisition only; not
converged. A from-scratch 20-update/2,621,440-step local-offline run on physical
GPU0 (`GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`, offline W&B id `3dz2fyz8`)
used the canonical V13 PPO settings. Update 5 reached mean hits 0.00202. Update
20 reached 0.01143 with 23 confirmed events from 1,345 physical contact edges
and 448 clearance crossings; confirmation fraction was 1.71%. Qvel exceedance
was zero, qacc exceedance was 0.0051%, qvel/qacc utilization was 8.18%/33.52%,
and neither racket-z termination fired. The run therefore passed its two
early-acquisition and safety criteria, but remained about 83 times below the
stage target 0.95 and intentionally stopped at the diagnostic cap. Directory:
`measured_qvel_rmp_vertical_v13_gpu0_seed20260821_20260821_local_offline_smoke20`.
Checkpoint/CSV/log SHA-256 values are respectively
`6a1763555fa0b6801621cb2353bdeb4b7493baf35e9e83fab6a2dcdfe22299ab`,
`6e39a0610faf417e37d64cbfa52ce51e554ef9d461845d012522ea634898a34c`, and
`d8c819fc3a631d5c2ba47b049cdd516007ce6343835422826e9be04261a787f4`.
Do not reinterpret this bounded smoke as stage convergence or start a formal
long run without multi-seed acquisition confirmation.

## Experimental Measured-QVEL RMP Bounded Reference V14

Status (2026-08-21): implemented, regression-tested and locally smoke-tested;
opt-in, from scratch, and not a released or hardware-covering profile. Profile
`goal_d455_measured_qvel_rmp_vertical_v14` inherits all 21 V13 stages, reward,
reset, gates, PPO settings, recovered RMP/RMP.md PD, XML and DR values. It
replaces only V13's stateless measured-q-relative lead with a persistent,
measured-state-bounded reference and appends the reference error to the actor:

```text
q_ref_next = q_ref + v_policy * dt
error      = clip(q_ref_next - q_measured, -error_max, +error_max)
q_ref_next = q_measured + error
q_target   = q_ref_next
```

For V14, `dt=0.005 s`, `error_max[j]=velocity_limit[j]*0.0175 s`, and
`actor_obs = V13_obs[0:50] || (q_ref-q_measured)[0:7]`. The implementation
also intersects the published reference with the hard joint-position limits.

The first saturated action therefore moves the reference by exactly one 5 ms
velocity integral, not by 17.5 ms. Repeated same-sign actions can retain
unexecuted intent until the bounded error reaches, per joint, 3.675, 3.675,
4.2, 4.2, 5.25, 5.25 and 5.25 degrees. Opposite-sign actions unwind the same
state. This supplies the continuity lost by re-anchoring every target to
measured q while preventing the unbounded hidden windup of legacy q_cmd. The
first 50 actor values remain index- and value-compatible with V13; the new
seven-value suffix makes the persistent reference state Markov-observable.
It is not `include_command_state`: no q_cmd, dq_cmd, actuator delay or
privileged DR value is exposed.

The mode is fail-closed. It requires measured feedback, direct QVEL-to-RMP,
the physical action limiter, a positive reference-error horizon, the 7-D
reference-error observation, and zero stateless target lead. It cannot be
enabled with command-feedback ablation or another actuator/planner path.
Historical profiles retain their original 50-D observations and control
outputs exactly. V14 checkpoints are 57-D and cannot resume or replace any
50-D V1--V13 or released checkpoint. A deployment adapter for this new state
and observation contract is not yet implemented, so V14 is simulation-only.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v14.sh`.
It preserves V13's formal online-W&B default, GPU0 UUID check, evidence audit,
from-scratch/no-resume rule, explicit incomplete-DR acknowledgement and local
offline-smoke override. No formal PPO session is started automatically.

Pre-registered control acceptance: a saturated first step must equal the 5 ms
QVEL increment; a constant measured state must allow accumulation to but never
beyond `qvel_limit * 17.5 ms`; each published reference must stay within that
envelope of the measured q sampled for that control tick; action reversal must
unwind the reference; and the 57-D prefix/suffix contract and invalid-stack
rejections must pass. The suffix intentionally reports the true posterior
`q_ref-q(t+1)` after RMP/MJX executes the tick, so it may differ from the
publication-time envelope by the joint's measured intra-tick displacement.
The paired learning probe uses V13's seed and PPO settings. By update 20 it should exceed
V13's mean hits `0.01143` and 23 confirmed events while retaining qvel/qacc
exceedance below `0.005`/`0.01` and rare racket workspace exits. A failure to
beat V13 is negative method evidence, not permission to widen the physical
velocity limit or silently increase the error horizon.

Bounded PPO outcome (2026-08-21): the same-seed 20-update/2,621,440-step
local-offline GPU0 probe passed the preregistered learning and safety targets
by a large margin. It used physical GPU0
`GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`, JAX `CudaDevice(id=0)`, offline
W&B id `w5168bb6`, and the canonical PPO settings. Mean hits rose from
0.03622 at update 1 to 0.27407 at update 5 and 0.99469 at update 20; the
corresponding V13 update-5/update-20 values were 0.00202/0.01143. Across the
probe V14 recorded 12,881 confirmed events from 22,421 physical contact edges
and 15,203 clearance crossings, versus V13's 23/1,345/448. Update 20 hit1 was
0.95324, mean return 1.2208, mean episode length 139.67 steps, KL 0.00451 and
explained variance 0.8298. Mean hit racket/ball horizontal speed was
0.1643/0.5014 m/s, mean apex relative height was 0.2973 m, and the recent
camera-visible/view-in-bounds/z-ideal fractions were 0.93/0.86/0.95. Qvel
exceedance remained zero; maximum qacc
exceedance was 0.00608, below the 0.01 gate; maximum qvel/qacc utilization was
18.58%/57.78%. Racket-low and anchor exits were zero, with the worst racket
workspace-exit rate only 0.000572. Update-20 termination rates were dominated
by ball-low 0.006630, followed by ball-high 0.000336, ball-y 0.000107, ball-x
0.000076 and racket-high 0.000046; truncation was zero. The update-20 instantaneous mean exceeded
the 0.95 stage target, but the 24-update convergence window was still 0.8915
when the deliberately shorter 20-update cap fired, so the launcher's nonzero
exit is an expected diagnostic-cap result and not a crash or formal stage
graduation.

Run directory:
`measured_qvel_rmp_vertical_v14_gpu0_seed20260821_20260821_local_offline_smoke20`.
The exact smoke control-source hashes for `mjx_juggle_env.py`, trainer and
launcher were respectively
`affe2d20cf23a62e140cbb8333324b90f2aaf17a158257d99acd43ff391ba797`,
`0eb51011a13b29a915878db1cda3095d3a58b975f95d448b10a48c9ab4ad2d36`, and
`137866768690e232d30d3802f9087a4890ae3d01ef4129a73249b026d05e7dc9`.
Checkpoint/CSV/log SHA-256 values are respectively
`80837a1379d6b94423fb9623c3c4405b2dfe358c7935642a618b4b51764a2197`,
`251c9539453eddc950f986b81d6661b92ab379c5bd3b9804180adf363e452a51`, and
`360634e515f9e58e51e121f2006ecdb31b03d9b1d7bf31f02b057d76d1222cfd`.
This is strong single-seed evidence that lost integration memory, rather than
reward alone or RMP alone, was the dominant V12/V13 acquisition bottleneck.
It is not yet multi-seed convergence or sim-to-real evidence.

The stateless-lead alternative remains calibratable, but the parameter should
be identified from the complete closed-loop transfer from published target to
measured joint/racket motion, not copied from the RMP output-delay setting.
RMP target/output filters, estimator, Kp/Kd/feedforward, simulated output
delay, low-level PD and joint inertia all contribute frequency-dependent gain
and phase. Use fixed-state step/chirp replay at the 200 Hz publication boundary
to estimate that transfer over the juggling stroke band, sweep the smallest
lead that restores the required racket velocity, and reject any value that
violates pointwise qvel/qacc/contact-quality gates. A pure delay estimate is
only an initialization for that sweep; `recovered_rmp_qvel_target_lead_s` also
scales position-error amplitude and is not a literal time predictor.

## Experimental Measured-QVEL RMP Per-Joint Bounded Reference V15

Status (2026-08-21): implemented, regression-tested and passed one same-seed
20-update GPU0 learning/safety probe; opt-in, from scratch, simulation-only,
and not proven better than V14. Profile
`goal_d455_measured_qvel_rmp_vertical_v15` and launcher
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v15.sh`
inherit V14's 21 stages, 57-D observation, reward, reset, gates, PPO, RMP/PD,
XML and DR contracts. The only control change is:

```text
q_ref_next = q_ref + v_policy * dt
error      = clip(q_ref_next - q_measured, -error_max, +error_max)
q_ref_next = q_measured + error
q_target   = q_ref_next

dt             = 0.005 s
error_max[j]   = velocity_limit[j] * t_i[j]
t_i[0:7]       = [18, 19, 19, 18, 18, 18, 18] ms
```

The scalar `recovered_rmp_qvel_reference_error_horizon_s` and seven-value
`..._per_joint` field are mutually exclusive; values must be finite, positive
and exactly seven long. This is a stateful anti-windup/authority bound. It is
dimensionally related to a stateless lead but dynamically different: the
reference adds only `v_policy*dt` each tick and retains/unwinds its error,
whereas a stateless lead recreates the full `v_policy*t` displacement from
measured q every tick. Therefore `t_i` and
`recovered_rmp_qvel_target_lead_s` must not be treated as interchangeable
parameters, and separately measured delay components must not be blindly
summed.

Delay evidence and preregistered screen (physical GPU0, 2026-08-21): the
canonical recovered-RMP configuration identifies an internal output delay of
`[18,19,19,18,18,18,18]` 1-kHz ticks; the real final-RMP-reference to encoder
feedback diagnostic reports `[13.10,13.10,14.60,13.35,12.35,12.30,12.55] ms`.
These measure different boundaries. A fixed-stroke screen used seven onset
lanes at each 2.5/3.7/4.0 g mass and required at least 4/7 hits at every mass,
zero qvel exceedance and qacc exceedance at most 0.01. Uniform 17.5 ms and the
RMP-output vector both passed with 13/21 total lanes and zero exceedance. The
encoder-feedback vector failed with 6/21. Adding the two vectors reached 21/21
but was rejected: maximum qacc utilization was 1.01497 and its qacc-exceed
fraction was 0.42857. The selected V15 vector is therefore the smallest tested
per-joint physical-delay initialization that passed the hit/safety contract;
it is not an identified inverse of the full closed loop. Artifact
`qref_horizon_screen_20260821_v1/screen.json` SHA-256 is
`7cc087ed7f930b2eac7be32a64a1ff04cf8467086397647842eeefd4c65d747d`.
The RMP-config/encoder-feedback evidence hashes are respectively
`bf440aada19681d09e54163d1610a0a543a963e29a6f1533485d0442ffe295c3`
and `d2890df2c95b374a00157524c960c98585245c97550b226b518ad88f5a6cb60b`.

Same-seed PPO outcome: V15 update 5 mean hits was 0.31312 versus V14's
0.27407. Update 20 reached 0.99489 versus 0.99469, hit1 0.96728 versus
0.95324, and accumulated 13,478 versus 12,881 confirmed events. V15/V14
update-20 returns were 1.1884/1.2208. V15 qvel exceedance remained zero;
qvel utilization was 0.19079, qacc exceedance 0.00719 and qacc utilization
0.59053, compared with V14's 0.18569/0.00608/0.57776. Racket-low and anchor
exits were zero and racket-high was `2.29e-5`. Thus V15 passes the same
acquisition and safety criteria but its tiny single-seed hit difference does
not establish superiority, while its acceleration margin is slightly worse.
Keep V14 as the preserved control and require multi-seed/dynamic closed-loop
evidence before promotion.

Run directory:
`measured_qvel_rmp_vertical_v15_gpu0_seed20260821_20260821_local_offline_smoke20`;
offline W&B id `0v654044`. The smoke source hashes for environment, trainer,
launcher and screen tool are respectively
`39e4926eaa1b3ca9b6697d21dfc389f01fcbb0a1ec70f5c8869429e96bde2226`,
`11433c2642ec7267d2cc1dce0e650f5acb58021e3423586b7f97461b6abb77f2`,
`d1ca534b198c3bb99e33e5d0a7ad0baf4abe324d5f49fe94fb5bbc6016faf9ab`
and `6eb47fba7366c7c893801e1f3bff4c37093abc837c94f13bcbf00eb5dabdf883`.
Checkpoint/CSV/log hashes are respectively
`ceb173c23706ef81696123f40d2ca50e6b5e7198ec2bfe7471bf731bd972920b`,
`824f290a23bc6b41e54ef8c5c60fb1312d4002a6260e9805cf3356db82a86191`
and `9b01e809d294ed00848d7ea421babb00a1c901c77b6d905f4a66ed92b67a0d4b`.
The nonzero launcher exit is expected because the 24-update convergence
window cannot fill before the deliberate 20-update cap; it is not a crash or
stage graduation.

## Experimental Measured-QVEL RMP In-View Coverage Continuation V16

Status (2026-08-22): stopped at the user's request after a long stage-12
plateau. Profile `goal_d455_measured_qvel_rmp_vertical_v16` is a
continuation-only successor to V15. Its only launcher is
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v16_resume_v15.sh`.

V16 preserves V15 stages 1--21 exactly, including names, environments,
rewards, gates, 57-D actor, asymmetric critic, bounded `q_ref`, RMP/PD, DR and
PPO settings. Existing stage 21 is already the polish stage:
`rmp15_launch19_final_consolidation`. The six harder stages below are appended
only after that consolidation:

| Stage | New difficulty axis | Target X/Y | Target hold probability/tail | Ball vxy mean/RMS gate | Racket vxy mean/RMS; angular gate |
| --- | --- | --- | --- | --- | --- |
| 22 `rmp16_inview_noise_support` | 90 Hz normal in-view noise | +/-0.090 / +/-0.070 m | 0 / 0 | 0.22 / 0.30 m/s | 0.080 / 0.130 m/s; 0.95 rad/s |
| 23 `rmp16_inview_ball_state_xy_bridge` | ball-state/XY bridge | +/-0.120 / +/-0.085 m | 0 / 0 | 0.20 / 0.28 m/s | 0.075 / 0.120 m/s; 0.90 rad/s |
| 24 `rmp16_inview_ball_state_xy_wide` | widest stable candidate XY | +/-0.150 / +/-0.100 m | 0 / 0 | 0.19 / 0.27 m/s | 0.070 / 0.115 m/s; 0.85 rad/s |
| 25 `rmp16_inview_measured_target_hold_bridge` | measured miss bridge | +/-0.150 / +/-0.100 m | 0.05 / 0.011 | 0.18 / 0.25 m/s | 0.070 / 0.110 m/s; 0.82 rad/s |
| 26 `rmp16_inview_measured_target_hold_joint` | full measured miss model | +/-0.150 / +/-0.100 m | 0.08 / 0.022, 2--9 ticks | 0.18 / 0.25 m/s | 0.070 / 0.110 m/s; 0.80 rad/s |
| 27 `rmp16_inview_wide_xy_stable_consolidation` | unchanged-domain stability polish | +/-0.150 / +/-0.100 m | 0.08 / 0.022, 2--9 ticks | 0.16 / 0.23 m/s | 0.065 / 0.105 m/s; 0.80 rad/s |

Stages 22--27 use fractional 90 Hz observations. Stage 22 starts from the
existing wide 7 mm/0.07 m/s noise distribution. Stages 23--27 use candidate
8 mm position and 0.08 m/s velocity noise, 7 mm per-episode position bias,
0.07/0.07/0.10 m/s velocity bias, +/-1.2% scale, and inherit V15's full
ball-mass/inertia/spin/contact and RMP/PD candidate DR. Wide XY and the
inherited vertical apex/cadence course jointly expand the observed ball-state
support; this is a curriculum mechanism, not proof that real componentwise
quantiles are covered.

Out-of-view prediction is intentionally not trained. Every appended stage
sets camera/view missing, refresh dropout, burst loss and coherent missing to
zero and terminates on the registered D455 view boundary. Target publication
holds are also kept at zero through the full large-XY stage, then introduced
as a separate joint difficulty axis. The late 8% base probability and 2.2%
2--9-tick tail retain the V2/V15 measured scheduler contract and remain before
RMP; post-RMP execution holds stay disabled.

The default source checkpoint is
`measured_qvel_rmp_vertical_v15_gpu0_seed20260821_20260821_formal1/mjx_curriculum_last.pkl`,
SHA-256
`69939ed6e4223184a688299181d4d6c87533b9bca3d198a6a24dac5f16bd1163`.
Its matching `curriculum_progress.csv` SHA-256 is
`bde16496afa0073ecbc4396f60647a41b2d0f8222f70b8a06d4ca581cf04fb5c`.
The checkpoint is at stage 12, `rmp15_launch10_workspace_wide`; the CSV has one
newer row, which `--resume-curriculum-state` discards by serialized checkpoint
step. The launcher pins both hashes, requires the explicit stage, preserves
actor/critic/Adam moments and convergence history, rejects GPU sharing or less
than 8 GiB available host memory, uses no XLA preallocation, and creates a new
run directory. The user explicitly authorized W&B run `9acnp70r` to continue;
its history already contains update 11 at step 268435456 while the checkpoint
contains update 10 at 268304384, so the launcher uses the W&B-only 131072-step
offset and starts new W&B history at 268566528. The checkpoint and CSV retain
their true steps. Trainer-side validation permits only V15/V16 checkpoints
with the exact 57-D bounded-reference configuration and prohibits optimizer or
critic resets.

The resumed V16 run is
`measured_qvel_rmp_vertical_v16_gpu0_seed20260821_20260821_resume_v15_inview_xy_hold`
on physical GPU0 and W&B `9acnp70r`. It was stopped after stage update 6567;
the last periodic checkpoint is update 6565 with SHA-256
`8dad0683c414dce0fa1a87a88636b3e9271d9d5e2cd92640640b075ecfd0cb98`.
Task performance had not collapsed: rolling hits were about 14--15 and the
performance gate often passed. Graduation was blocked by behavior quality.
Mean hit racket XY speed remained about 0.107--0.114 m/s against 0.10 m/s,
and full hit racket angular speed remained near 1.29 rad/s against 1.20 rad/s;
the angular gate passed zero rolling windows. Mean apex was about 0.221 m,
inside but near the 0.218--0.248 m band. PPO diagnostics remained stable
(KL about 0.0103, explained variance about 0.87, zero qvel exceedance and
roughly 0.6% qacc exceedance), so this is recorded as a reward/gate alignment
plateau rather than a QVEL/RMP hit-acquisition failure.

Acceptance remains experimental. The candidate must pass the listed hit,
episode-length, hit1/hit3/hit12, view, truncation, apex/cadence, qvel/qacc and
horizontal/angular stability gates without increasing failures. Before any
hardware-coverage claim, add paired real in-view noise and ball-state
componentwise evidence, the missing new-ball outcome trials, and formal
pointwise RMP/PD encoder-feedback coverage. V16 is simulation-only and does
not change the deployment adapter.

## Experimental Measured-QVEL RMP Motion-Quality/Apex Continuation V17

Status (2026-08-22): implemented and regression-tested. One initialization-only
launch was stopped before its first PPO update so the full-course lineage could
be corrected; it produced no checkpoint or curriculum-progress row. Profile
`goal_d455_measured_qvel_rmp_vertical_v17` preserves V16's 57-D actor,
asymmetric critic, measured-feedback bounded `q_ref`, per-joint 18/19 ms error
horizons, recovered `recovered_rmp_rmpmd_v2` RMP/PD stack, physics, reset
distributions, observation contract and candidate DR. It is a new
reward/curriculum identity and does not modify or reinterpret V16.

The parent is the V16 periodic archive
`archive_13_rmp15_launch10_workspace_wide_update_6150.pkl`, SHA-256
`5ee9f671a379f09629939f25bf5f2df36051160d2e483af7098044b34450e82f`,
from
`measured_qvel_rmp_vertical_v16_gpu0_seed20260821_20260821_resume_v15_inview_xy_hold`.
It contains `step=1073086464`, serialized trainer stage index 12,
`stage_update=6150`, `global_update=8187`, actor/critic parameters and Adam
moments. Among archived checkpoints whose performance gate passed, it
minimized the maximum normalized launch10 racket-XY/angular gate overshoot;
its logged hits/apex/qacc-exceedance were about 14.67/0.2213/0.00576, with
0.1065 m/s mean racket XY and 1.2855 rad/s full angular speed. This is a
predeclared checkpoint selection rule, not a best-return selection.

A deterministic 64-environment screen from that checkpoint produced 15.31
mean hits, 19 maximum hits and 0.953 full-episode rate. Its CSV is
`video_stage12_archive6150_seed20260822/screen64.csv`, SHA-256
`ca3d51982d0b3bf3732c4c9ea205e107f0abdafd9bea3220be4ec0b71766ce31`.
The representative env 41 completed all 1200 steps with 16 hits. The 1x video
is `archive6150_env41_16hits_1x.mp4` (SHA-256
`ba098086da06bb9172ff646e827d74a8d2ecd6523d501a7260d6f545364f43e6`)
in the same evidence directory; the 2x-slow version has SHA-256
`092d0cf62fffcbadf94c27a8f7a974e1b2ca10bdd09e90f16be9a1ae67fcfa01`.

V17 is a full-course profile built from a one-to-one copy of all 27 V16 stages,
then modified only at and after the selected continuation point. V16 stages
1--11 remain exact, unchanged V17 stages 1--11 and are never replayed by this
continuation; V16 stage 12 remains V17 stage 12 and becomes the
first repair bridge; two same-domain repair stages are inserted; all V16 stages
13--27 remain present in order as V17 stages 15--29. Zero-based resume index 11
therefore starts a fresh convergence window under the changed rewards while
restoring the full parent train state:

```text
00 rmp15_discovery00_vertical_launch_no_teacher
01 rmp15_launch00_acquisition
02 rmp15_launch01_local_workspace
03 rmp15_launch02_workspace
04 rmp15_launch03_ball_dynamics_mild
05 rmp15_launch04_contact_dynamics_mild
06 rmp15_rmp05_rmp_pd_mild
07 rmp15_launch06_racket_geometry_mild
08 rmp15_launch07_observation_calibration_mild
09 rmp15_launch08_single_dropout_preview
10 rmp15_launch09_camera_missing_mild
11 rmp17_motion_quality_bridge_135_012
12 rmp17_motion_quality_bridge_127_011
13 rmp17_motion_quality_commit_120_010
14 rmp17_launch11_ball_dynamics_wide
15 rmp17_launch12_contact_dynamics_wide
16 rmp17_rmp13_rmp_pd_coverage
17 rmp17_launch14_racket_geometry_wide
18 rmp17_launch15_observation_calibration_micro_bridge
19 rmp17_launch16_observation_calibration_bridge
20 rmp17_launch17_observation_calibration_wide
21 rmp17_launch18_camera_missing_wide
22 rmp17_launch19_final_consolidation
23 rmp17_inview_noise_support
24 rmp17_inview_ball_state_xy_bridge
25 rmp17_inview_ball_state_xy_wide
26 rmp17_inview_measured_target_hold_bridge
27 rmp17_inview_measured_target_hold_joint
28 rmp17_inview_wide_xy_stable_consolidation
```

Stages 0--10 preserve the corresponding V16 stage objects exactly, including
their names, environment, rewards, gates, notes and historical update limits;
the continuation does not execute them. Stages 11--13
repeat the unchanged V16 launch10 domain. Their mean hit-racket
XY/full-angular/apex-lower gates are `0.12/1.35/0.220`,
`0.11/1.27/0.223`, then the original strict `0.10/1.20` motion limits with a
raised 0.225 m apex floor; all retain the 0.248 m apex ceiling. Hit racket-XY
and angular penalty weights rise from the V16 source value 0.50 to
`0.75/1.00/1.25`. A bounded positive low-angular confirmed-hit reward uses
weights `0.35/0.55/0.75`, targets `1.25/1.18/1.10 rad/s`, and sigmas
`0.35/0.30/0.30 rad/s`. A bounded target-apex reward uses the same weight
ladder around the unchanged 0.23 m target with 0.030 m sigma. Stages 14--28
retain the committed 1.25 motion penalties, 0.75 angular/apex rewards and at
least the 0.225--0.248 m apex band while otherwise mapping one-to-one to the
remaining V16 domains and gates.

V17 stages 11--28 have no profile update cap, the launcher supplies no
per-stage update limit, and it never sets `--allow-unconverged-advance`;
active stages advance only after the complete rolling and block-validation
gates pass. The continuation uses 1024 environments,
128 steps, minibatch 16384, four epochs, LR `1.5e-4`, gamma `0.9995`, GAE
`0.99`, clip `0.15`, target KL `0.008`, entropy `0.0002`, and a 24-update
convergence window. These less aggressive LR/clip/KL values are intended to
adapt the mature hit policy without rapidly erasing it.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v17_resume_v16.sh`.
It pins the parent and selection-screen hashes, requires the exact source
profile/stage/57-D contract, preserves actor/critic/Adam, prohibits optimizer
or critic reset, and intentionally prohibits `--resume-curriculum-state`
because V17 rewards and stage identities differ. It uses seed 20260822, a new
output directory, and the user explicitly authorized it to append to W&B
`9acnp70r`. The launcher uses W&B resume mode `must` and a W&B-only step offset
of `54788096`: the last V16 point may be step 1127874560, while the first new
V17 point becomes 1128005632. The source checkpoint retains its true
1073086464 step and optimizer state. GPU0 UUID/ownership, >=8 GiB host-memory,
XLA no-preallocation and 78 C safe-stop checks remain mandatory.

Acceptance requires the complete existing task, view, survival, apex,
cadence, qvel/qacc, horizontal and angular rolling gates plus block validation
at every stage. Lack of convergence is not permission to force advancement.
V17 is experimental and simulation-only; the paired real in-view camera
audit, new-ball outcome trials and formal RMP/PD feedback coverage remain
unresolved, and no deployment adapter is authorized by this profile.

## Experimental Measured-QVEL RMP Bounded-Hit/Apex-Survival V18

Status (2026-08-22): implemented and regression-tested after the user stopped
V17 Stage 14. V17 reached the strict angular gate but remained stalled for
1534 updates: its rolling apex stayed near 0.220--0.221 m against the 0.225 m
floor while hits rose to about 15.1 per 1200-step episode. This exposed a
reward mismatch, not a control-path failure. V17 still paid an unconditional
`hit_count_floor_reward_weight=1.5` and `center_flat_hit_reward_weight=2.4`
at every confirmed hit with `hit_reward_cap_mode=off`, while dense post-hit
survival had weight 1.4 and the existing low-apex loss did not activate until
0.195 m. The policy could therefore improve return by shortening the cycle
and reducing angular speed while sacrificing a few millimetres of apex.

Profile `goal_d455_measured_qvel_rmp_vertical_v18` preserves the full 29-stage
V17 course. Stages 1--13 are exact V17 objects. Stages 14--29 retain their
control, 57-D observation, RMP/PD, physics, reset, DR, task, view, safety,
motion-quality and apex gates, but use these reward/cadence changes:

```text
hit_reward_cap_mode             = fixed
hit_reward_count_cap            = 14
post_hit_survival_reward_weight = 2.4
low_hit_apex_margin             = 0.005 m
low_hit_penalty_weight          = 10000 m^-2
hit_cadence_target_interval     = 0.43 s  (unchanged)
hit_cadence_reward_weight       = 0.50
hit_cadence_sigma               = 0.060 s
graduation hit interval         = 0.40--0.50 s
```

Because `target_height=0.230 m`, the 0.005 m low-apex margin activates the
one-sided loss exactly below 0.225 m. A 4 mm deficit costs 0.16 reward at a
rewarded confirmed hit; there is no incentive to increase apex above the
existing 0.23 m Gaussian target, and the 0.248 m upper gate remains. The hit
cap bounds event credit but does not terminate the episode, alter physical hit
accounting, penalize an extra necessary contact, or add a maximum-hit gate.
The desired 13--14 contacts per 1200 steps are enforced indirectly by the
physical apex/cadence band. Longevity is judged primarily by mean episode
length (`conv_len`), full-horizon truncation, hit1/hit3/hit12, view and safety.

The parent is the V17 transition checkpoint
`13_rmp17_motion_quality_bridge_127_011.pkl`, SHA-256
`fe1f2a66becf5fe9d7056112d534961b10e6ae769e34d66231e951f0976a7936`,
at true step `1196032000`. It is the checkpoint saved immediately after the
second bridge passed and on entry to Stage 14, before the long V17 reward
tradeoff. V18 resumes actor, critic and Adam moments at zero-based stage 13
(`rmp18_motion_quality_commit_120_010`) with fresh convergence history. It
rejects another source profile/step, curriculum-history resume, optimizer or
critic reset, and any non-57-D bounded-reference contract.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v18_resume_v17.sh`.
PPO remains V17's LR `1.5e-4`, clip `0.15`, target KL `0.008`, 1024 x 128
rollouts and four epochs. No stage/update cap is set. The user explicitly
authorized continued logging to W&B `9acnp70r`; `resume=must` and the
W&B-only offset `255852544` put the first V18 point one rollout after V17's
last logged point. The checkpoint's true step remains unchanged. The launcher
retains GPU0 UUID/ownership, >=8 GiB host-memory, no-preallocation and 78 C
safety checks. V18 is experimental and simulation-only and inherits all
camera, new-ball outcome and formal RMP/PD coverage blockers.

## Experimental Measured-QVEL RMP True-Cadence/VXY Bridge V24

Status (2026-08-22): accepted for formal continuation after one frozen-policy
audit, four bounded negative shaping experiments, regression tests, and a
30-update GPU0 acceptance run. V24 preserves the V18 control stack, 57-D
observation, bounded measured-q reference, RMP/PD, physics, DR, resets, apex
and survival rewards, fixed 14-hit event-credit cap, PPO settings and final
quality contracts.

The root curriculum bug was a cadence-unit mismatch. The old graduation value
used episode duration divided by hit count, which includes first-contact and
terminal slack and is not the interval between adjacent physical contacts.
Frozen stochastic validation of the selected V18 best checkpoint used 128
environments for one 1200-step episode each. Among 115 full episodes it found:

```text
mean hits                             15.8957
true adjacent count-gated interval    0.346963 s
mean apex relative to anchor           0.242426 m
mean hit-racket vxy                    0.102007 m/s
mean first observed post-confirm lift  0.085574 m
```

The last value is remaining rise after hit confirmation, not the complete
contact-to-apex height. Do not equate it with the 0.23 m anchor-relative apex.
Nor should the 0.23 m apex be converted to a same-height ballistic period,
because the moving racket generally makes the next contact at a different
height.

V19 enabled the true metric but used a 0.390--0.460 s target with a sparse
10-weight early-contact loss; after 40 updates its recent interval remained
`0.3466 s`. V20 increased that loss to 60 and the cadence bonus to 1.5; the
period still did not move. V21 instead constrained racket z to +/-0.04 m; it
also left the period unchanged. V23 multiplied positive hit quality by a
bounded low-racket-vxy score; it did not improve the 24-update vxy window and
reduced return. These are recorded negative results, not resume candidates.
All bounded runs are under
`pingpong_controller/outputs/rl_sim/v19_cadence_gate_experiments_20260822/`.

V22 established the accepted cadence semantics, and V24 retains them:

```text
hit_interval_gate_source              = counted_hit_event
graduation counted-hit interval        = 0.300--0.400 s
hit_cadence_reward_weight              = 0
hit_min_interval_penalty_weight        = 0
hit_max_interval_penalty_weight        = 0
hit_reward_count_cap                   = 14  (unchanged from V18)
post_hit_survival_reward_weight        = 2.4 (unchanged from V18)
apex graduation                        = 0.225--0.248 m (unchanged)
```

V22 then showed that the only remaining strict failure was mean hit-racket
vxy: `0.10144 m/s` against the first-stage `0.100 m/s` gate. V18's recurrent
value had stayed around `0.1036 m/s` for hundreds of updates, and the former
next-stage gate jumped immediately to `0.0967 m/s`. V24 changes no reward. It
only makes this remaining 16-stage curriculum bridge feasible while retaining
the final contract:

```text
mean hit-racket-vxy gate (m/s):
0.105, 0.105, 0.103, 0.101, 0.099, 0.097, 0.095, 0.092,
0.089, 0.086, 0.083, 0.080, 0.077, 0.073, 0.069, 0.065
```

The 30-update V24 GPU0 acceptance run ended with no failed strict gate: hits
`14.6351`, episode-length fraction `0.9308`, true interval `0.34589 s`,
mean/RMS hit-racket vxy `0.10307/0.11959 m/s`, racket angular speed
`1.09190 rad/s`, apex `0.24453 m`, qvel exceedance zero and last-12-update
qacc exceedance fraction `0.00476`. Mean PPO KL was `0.00627`, below the
`0.008` target. The experiment stopped only because its 30-update cap was
below the unchanged Stage-14 minimum 140 updates.

The formal parent is V18 `mjx_curriculum_best.pkl`, update 372, true step
`1244790784`, SHA-256
`0c705462508a0c26a3b3f92c1003adcc442240a0e7904f1c42734e98d5472955`.
Do not use V18 update 400 or any V19--V24 bounded trial checkpoint. Canonical
launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v24_resume_v18.sh`.
It has no update cap, keeps LR `1.5e-4`, clip `0.15`, target KL `0.008`, 1024 x
128 rollouts, four epochs, Stage-14 minimum 140 updates, 20-update hold and
block validation. The user explicitly authorized W&B `9acnp70r`; `resume=must`
and W&B-only offset `260046848` place the first V24 point one rollout after
the unsaved V18 update-404 W&B tail. V24 remains experimental and
simulation-only and inherits all unresolved hardware-evidence blockers.

## Experimental Measured-QVEL RMP Absolute-Apex/Lift V25

Status (2026-08-22): bounded acceptance passed; opt-in formal continuation.
V24 is preserved. V25 resumes the safely stopped V24
`mjx_curriculum_last.pkl` at update 565, true step `1318846464`, SHA-256
`82ff1b38f199043c8fefcda282a539d89caa418e3c70e58cc94f54f11bc8076b`.
The source keeps 57-D measured bounded-reference observations, recovered RMP,
PD, actor, critic and Adam moments. V25 starts fresh curriculum/convergence
history at `rmp25_apex_abs_bridge_134` because reward and gate semantics
change.

The stopped V24 curve exposed a coordinate and reward mismatch. Its
`mean_hit_apex_rel_height_m ~= 0.255 m` was relative to the episode racket
anchor; `mean_hit_ball_z ~= 1.196 m` and
`mean_hit_apex_lift_m ~= 0.107 m` imply only about `1.303 m` absolute apex.
The inherited one-sided low-apex loss used weight `10000 m^-2` below the
relative `0.225 m` floor, while the upper event and dense barriers began only
near relative `0.265 m`. In the V24 tail, reducing this low loss yielded about
an order of magnitude more per-step reward than was lost from the symmetric
target term, so PPO rationally raised apex while remaining unable to satisfy
the `<=0.248 m` gate.

V25 replaces that anchor-relative loss and aligns reward and gates in world-z
and physical contact-to-apex lift:

```text
bridge absolute apex targets (m)     1.34 -> 1.37 -> 1.40
bridge mean apex bands (m)           1.30--1.37, 1.32--1.40, 1.36--1.43
bridge lift targets (m)              0.18, 0.20, 0.22
bridge mean lift bands (m)           0.159--0.205, 0.175--0.230, 0.190--0.260
bridge max mean ball contact z (m)   1.160, 1.180, 1.200
bridge max mean racket contact z (m) 1.140, 1.160, 1.180
true ball-z termination ceiling (m)  1.45
absolute-apex target reward          weight 3, sigma 0.040/0.035/0.030 m
lift target reward                   weight 3, sigma 0.035/0.030/0.030 m
two-sided apex barriers              +/-0.030 m, weight 250 m^-2
absolute lower-band loss             weight 2500/3500/5000 m^-2
two-sided lift band loss             weight 2500 m^-2
contact-height penalty               weight 400 m^-2
```

The `1.45 m` ceiling applies to the ball, not the racket. V25 separately
caches ball and racket z at the physical contact edge. Historical curricula
keep their delayed-confirmation height semantics through the default
`hit_contact_height_measurement_mode=confirmation`; only V25 selects
`contact_edge`. This also makes `mean_hit_apex_lift_m` a complete
contact-to-apex rise rather than only the post-confirmation remainder.
V25 alone decouples its apex target credit from the combined lateral/pose
quality multiplier, so reducing horizontal or angular motion cannot masquerade
as improving apex. The apex/lift credits remain limited to rewardable counted
hits and therefore retain the 14-event cap. Historical profiles keep the
coupled default and zero lift reward/loss.

The final bounded acceptance used 512 environments and 24 updates from the
pinned source. Its late four-update window improved absolute apex/lift from
`1.30162/0.15914 m` in the first stable window to `1.30231/0.15950 m` while
keeping ball/racket physical contact z at `1.14281/1.12290 m`. Late hits were
`14.30`, hit-racket vxy `0.10356 m/s`, qvel exceedance zero, qacc exceedance
`0.00554`, and KL `0.00655`. The first bridge's `0.159 m` lift gate is set from
that measured source-compatible window; it is not a claim that the final
`0.18--0.26 m` lift or `1.37--1.43 m` apex contract has been learned. Rejected
screens retained the coupled quality shortcut, changed only positive apex
weight, or used oversized height loss without a lift contract; none of their
checkpoints may be resumed. Full evidence is recorded under
`outputs/rl_sim/v25_absolute_apex_experiments_20260822/`.

Real record_new3/4/5 contact evidence, after excluding obviously unusable
height/velocity samples, had median contact ball z about `1.191 m`, median
absolute apex about `1.339 m`, and median adjacent-contact interval about
`0.42 s` across 47 recordings. It is supporting scale evidence, not a formal
camera-coverage population. Ballistically,
`T=sqrt(8*(apex-hit_z)/g)`: contact around `1.18--1.20 m` and apex
`1.37--1.43 m` imply roughly `0.37--0.45 s` of same-height flight and retain
at least 20 mm below the user's preferred `1.45 m` ball ceiling. V25 therefore
uses counted-event cadence gates `0.31--0.43`, `0.34--0.45`, and
`0.36--0.46 s` across the bridges, then `0.38--0.46 s` downstream. Cadence
reward and interval losses stay zero; 14-hit event credit remains capped and
full survival remains desirable.

V25 preserves V24 stages 1--13 exactly, inserts the three apex lessons, then
retains every V24 stage 15--29 in order as V25 stages 17--31 with the committed
physical apex contract. There is no update cap. Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v25_resume_v24.sh`.
It keeps PPO LR `1.5e-4`, clip `0.15`, target KL `0.008`, 1024 x 128 rollouts,
four epochs, GPU0 ownership/memory/temperature checks, W&B `9acnp70r` with
`resume=must`, and W&B-only offset `260571136`. V25 remains experimental and
simulation-only and inherits all unresolved real-camera, new-ball-outcome,
formal RMP/PD coverage and deployment-adapter blockers.

## Experimental Measured-QVEL RMP Survival-Balanced V26

Status (2026-08-23): accepted bounded direction and PPO configuration; opt-in
formal continuation. V25 is preserved. V26 resumes V25 stage-15 update-100,
true step `1345060864`, checkpoint
`archive_16_rmp25_apex_abs_bridge_137_update_0100.pkl`, SHA-256
`c5f9a6997ddef0df08558ac3dbaf1b226758ab35dc6ef2b64ce6315a2f2c6b00`.
It preserves the 57-D bounded measured-q reference, actor, critic, Adam,
recovered RMP/PD, physics, DR and reset domain, but starts a fresh V26
curriculum window because reward and gate semantics change.

V25 was stopped after stage-15 update 1030 because its height objective was
not improving task longevity. Representative early-to-late windows were:

```text
full episode rate          0.820 -> 0.743
conv_len                   0.898 -> 0.865
mean hits                  13.59 -> 12.49
absolute apex (m)          1.309 -> 1.337
contact-to-apex lift (m)   0.163 -> 0.181
```

The direct cause was reward-scale/coverage mismatch. Confirmed height credit
remained active while the historical `post_hit_survival` term was masked out
during part of the falling flight. The resulting term is opt-in and
count-gated:

```text
post_first_hit_alive = weight * (counted_hit_count > 0)
```

It does not reward a zero-hit episode, change termination, or uncap positive
hit events. Historical profiles keep weight zero. V26 uses:

```text
lesson                         recovery   balanced   commit
post-first-hit alive weight       3.0        3.5       4.0
minimum conv_len                  0.90       0.92      0.94
minimum full-episode rate         0.82       0.84      0.87
minimum mean hits                 13.0       13.0      13.0
max mean hit-racket vxy (m/s)    0.105      0.103     0.100
max full angular speed (rad/s)    1.27       1.23      1.20
minimum absolute apex (m)         1.30       1.31      1.32
maximum absolute apex (m)         1.37       1.37      1.37
minimum lift (m)                 0.160      0.165     0.170
maximum lift (m)                 0.205      0.205     0.205
adjacent counted cadence (s)   0.35--0.42 for all three
```

All three target `1.34 m` absolute apex and `0.18 m` lift. Symmetric height
and lift target rewards use weight `0.75`; two-sided apex and low-height
barriers use `150 m^-2`, lift uses `500 m^-2`, true ball-z stays below
`1.45 m`, and mean physical-edge ball/racket contact z remains below
`1.18/1.16 m`. Racket-z, lateral and full-angular penalties tighten across
the lessons. Hit credit remains capped at 14; cadence stays a health gate with
zero cadence reward and interval loss. Downstream stages keep apex/lift bands
`1.32--1.38/0.17--0.21 m`, cadence `0.36--0.42 s`, and at least
`0.90 conv_len`.

Four same-source 512-env, 24-update trials isolated PPO behavior and reward
scale. LR `1.5e-4` with four epochs accepted only 5/24 complete updates; LR
`1.0e-4` with four epochs accepted 11/24; stronger survival at LR `7.5e-5`
still accepted only 2/24. In each rejected setting, rollout-wide exact KL
slightly exceeded `0.008` and transactionally rolled the update back. The
accepted configuration uses LR `1.0e-4` and three epochs:

```text
metric                         updates 19--21   updates 22--24
conv_len                            0.89734          0.94368
full episode rate                   0.81987          0.87842
mean hits                          13.44345         14.14077
counted-hit interval (s)            0.36534          0.36529
absolute apex (m)                   1.31139          1.31071
lift (m)                            0.16712          0.16795
mean hit-racket vxy (m/s)           0.09907          0.09794
full racket angular speed (rad/s)   1.27700          1.26536
qvel exceedance                     0                0
qacc exceedance                     0.00627          0.00616
racket-too-high/step                0.000092         0.000036
ball in-view fraction               0.96837          0.97039
```

The last-three exact KL averaged `0.00605`, below the unchanged `0.008` hard
limit. Bounded caps were experiment-only. Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v26_resume_v25.sh`.
Formal training uses 1024 environments, 128 steps, minibatch 16384, LR
`1.0e-4`, three epochs, clip `0.15`, target KL `0.008`, no update cap and block
validation. It resumes W&B `9acnp70r` with `resume=must` and W&B-only offset
`382468096`. The first new W&B point follows the stopped V25 tail by one
131072-step rollout; true checkpoint time is unchanged. V26 remains
experimental, simulation-only, and inherits every real-camera, replacement
ball, formal RMP/PD coverage and deployment blocker.

## Experimental Measured-QVEL RMP Range-Aligned Reward V27

Status (2026-08-23): accepted for uncapped GPU0 continuation after a
formal-shape 1024-env bounded trial. V25 and V26 remain immutable. V27 resumes
only V25 stage index 15, `rmp25_apex_abs_bridge_137`, update 150 at true step
`1351614464` from
`archive_16_rmp25_apex_abs_bridge_137_update_0150.pkl`, SHA-256
`422054bf0833d63298ba4d89fbb233bf5d2154ef5e7d854dd433667a53a013ea`.
It restores the 57-D bounded measured-q reference, actor, critic and Adam
moments, but starts a fresh V27 convergence window at
`rmp27_survival_height_rebalance_134` because reward and gate semantics
change. Curriculum-history resume, optimizer/critic reset, another source
profile/stage/update/step, and a non-57-D checkpoint are rejected.

The source was selected before modification from the archived population,
not from a V27 result:

```text
metric                                  V25 update 150 source
conv_len                                          0.9053
full episode rate                                 0.8318
mean hits                                        13.5450
counted-hit interval (s)                          0.3648
absolute apex (m)                                 1.31447
contact-to-apex lift (m)                          0.16689
mean hit-racket vxy (m/s)                         0.09976
full racket angular speed (rad/s)                 1.26871
qvel/qacc exceedance                         0 / 0.00629
```

The stopped V26 update-50 checkpoint was rejected as the source despite its
slightly stronger longevity: its apex/lift were only `1.30447/0.16375 m`, and
multiple bounded probes could not leave that low-height basin. The V26 reward
audit also found a scale mismatch. Per-step logged apex/lift target credit was
only about `0.0038/0.0051`, while survival, contact and already-passing motion
terms were individually comparable or larger. The V26 hit-racket-vxy soft
limit was about `0.0945 m/s` against a `0.105 m/s` gate and the angular soft
limit about `0.864 rad/s` against a `1.27 rad/s` gate. PPO could therefore
earn more by suppressing motion beyond the gate than by correcting flight
height.

V27 keeps longevity primary but makes every bounded flight quantity point
toward an interval interior. For absolute apex `a` and lift `l`, the positive
scores are

```text
R_apex = w_a * exp(-0.5 * ((a - 1.34) / 0.035)^2)
R_lift = w_l * exp(-0.5 * ((l - 0.18) / 0.030)^2)
```

and their two-sided dead-band losses are proportional to
`max(0, abs(a-1.34)-0.015)^2` and
`max(0, abs(l-0.18)-0.008)^2`. A paired lower-apex guard and upper-apex soft
guard remain for safety. Counted cadence uses a much smaller symmetric
Gaussian score centred at `0.385/0.385/0.390 s`; minimum/maximum interval
penalties stay zero. Contact heights, vxy and angular speed only have upper
graduation limits, so their losses remain upper-only. Their soft limits are
placed just inside the active gates. This avoids creating an artificial
incentive to drive a passing motion metric continuously toward zero.

The first three V27 lessons are:

```text
lesson                           rebalance   centred   commit
post-first-hit alive weight           3.0       3.5      4.0
apex/lift target weight            2.0/2.0 2.25/2.25  2.5/2.5
apex dead-band weight (m^-2)          1000      1200     1400
lift dead-band weight (m^-2)          2000      2400     2800
cadence reward weight                 0.10      0.12     0.15
minimum conv_len                       0.90      0.92     0.94
minimum full rate                      0.82      0.84     0.87
apex band (m)                     1.30-1.37 1.31-1.37 1.32-1.37
lift band (m)                   0.160-0.205 0.165-0.205 0.170-0.205
counted cadence (s)               0.35-0.42 0.35-0.42 0.36-0.42
max mean hit-racket vxy (m/s)          0.105     0.103    0.100
max full angular speed (rad/s)          1.27      1.23     1.20
```

Mean hits and conditional mean hits retain a minimum of 13, positive event
credit remains capped at 14, and extra physical recovery contacts are not a
failure. Later V26 domains remain in order with at least `0.90` conv_len,
apex `1.32--1.38 m`, lift `0.17--0.21 m`, counted cadence `0.36--0.42 s`, and
their existing lateral-motion progression. The profile has 32 stages and no
update cap.

The final 1024-env, 30-update acceptance was run with the formal PPO shape.
Its last-12 rolling means were:

```text
metric                                  source       V27 late-12
conv_len                               0.9053          0.90958
full episode rate                      0.8318          0.83385
mean hits                             13.5450         13.60364
counted-hit interval (s)               0.3648          0.36307
absolute apex (m)                      1.31447         1.31321
contact-to-apex lift (m)               0.16689         0.16547
mean hit-racket vxy (m/s)              0.09976         0.10017
full racket angular speed (rad/s)      1.26871         1.25852
qvel exceedance                        0               0
qacc exceedance                        0.00629         0.00647
exact KL                               0.00669         0.00611
```

Every strict first-stage gate passed. Survival improved slightly and apex/lift
changed only `-1.3/-1.4 mm`, which is treated as no material short-window
regression, not as proof of an upward height trend. The next two lower-gate
steps are deliberately stricter: promotion requires the long run to move
apex/lift inward while preserving longevity. Negative trials, reward-term
magnitudes and the complete acceptance command are recorded in
`outputs/rl_sim/v27_range_reward_experiments_20260823/V27_RANGE_ALIGNED_REWARD_REPORT.md`.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v27_resume_v25.sh`.
Formal training uses 1024 environments, 128 steps, minibatch 16384, LR
`1.0e-4`, three epochs, clip `0.15`, target KL `0.008`, block validation, no
update cap, and the existing host/GPU safety guards. It continues W&B
`9acnp70r` with `resume=must` and W&B-only offset `430833664`, placing the
first V27 point at `1782579200` while leaving true optimizer time unchanged.
V27 remains experimental, simulation-only, and inherits every real-camera,
replacement-ball, formal RMP/PD coverage and deployment blocker.

## Experimental Measured-QVEL RMP All-Tail Survival-Primary V28

Status (2026-08-23): accepted for uncapped GPU0 continuation after two paired
formal-shape 1024-env trials. V27 remains immutable. V28 resumes only V27's
stage-16 `mjx_curriculum_best.pkl` at update 4112, true step `1906966528`,
SHA-256
`a2e0f4efeb531defca3370d983c8075a70ec9176efd06a60aa5ed6319364012c`.
The selected source restores the 57-D actor, 221-D critic, Adam moments and
bounded measured-q reference but starts fresh V28 curriculum/convergence
history at `rmp28_survival_primary_physics_band`. Its frozen metrics were:

```text
conv_len                              0.94310
full episode rate                     0.90665
mean hits                            14.03722
adjacent counted-hit interval (s)     0.36917
absolute apex (m)                     1.30492
contact-to-apex lift (m)              0.16953
mean hit-racket vxy (m/s)             0.08200
full racket angular speed (rad/s)     1.34606
```

V27 was safely stopped after stage update 4264, having spent more than 4,200
updates in the same stage. Its late reward audit showed a structural conflict:

```text
reward per simulator step             V27 late-128
hit-apex-lift penalty                    -0.01921
hit-height penalty                       -0.01868
post-first-hit alive                     +0.01552
post-hit geometric survival              +0.00688
```

The two narrow height losses outweighed direct plus geometric survival even
though the measured flight was self-consistent. At contact z about 1.135 m,
lift `0.171 m` gives apex about `1.306 m` and ballistic same-height period
`sqrt(8*0.171/g)=0.374 s`, matching the measured roughly 0.370 s cadence.
The old simultaneous apex/lift targets `1.34/0.18 m` instead imply mutually
different contact heights: apex 1.34 at contact 1.135 needs lift 0.205 m,
while lift 0.18 gives apex only 1.315 m.

V28 preserves V27 stages 1--15 exactly and keeps every stage-16--32 domain
transition in order. It replaces reward and motion/height/survival gates for
the complete remaining tail, not only the current stage:

```text
direct post-first-hit alive weight                  5.0
absolute apex target / sigma (m)             1.32 / 0.050
lift target / sigma (m)                     0.175 / 0.040
bounded apex/lift reward weights                 0.75 / 0.75
mean apex band (m)                              1.28--1.39
mean lift band (m)                            0.145--0.220
adjacent counted cadence band (s)                0.33--0.44
apex/lift outside-band loss (m^-2)               100 / 100
duplicate one-sided low-apex loss                       0
true ball-z ceiling (m)                                1.45
max mean ball/racket contact z (m)                1.18/1.16
max mean/RMS hit-racket vxy (m/s)               0.120/0.170
full-angular soft target / gate (rad/s)           1.25/1.40
minimum mean and conditional hits                       13
positive event-credit cap                               14
minimum tail conv_len/full                         0.90/0.82
resume lesson conv_len/full                        0.93/0.88
unchanged-domain commit conv_len/full              0.94/0.89
final consolidation conv_len/full                  0.92/0.84
```

The height/lift loss dead bands cover the complete mean gate intervals, so
healthy flights have no barrier loss; bounded symmetric rewards still pull
gently toward the interior. Cadence is a reward-free health measurement.
Racket motion uses only mild upper-only penalties and bounded angular credit;
the former unsupported downstream ladder to `0.065 m/s` mean vxy and
`0.80 rad/s` full angular speed is removed from every later V28 stage. Ball
vxy, view, safety, domain, reset and DR gates remain stage-specific and
unchanged. There is no maximum-hit failure and no update cap.

Two same-seed trials used the production 1024 x 128 rollout and PPO shape,
differing only in LR. Their late-12 rolling means were:

```text
metric                         parent      LR 1.0e-4    LR 1.25e-4
conv_len                       0.94310       0.95201       0.93915
full episode rate              0.90665       0.92818       0.89956
mean hits                     14.03722      14.19801      13.91902
counted-hit interval (s)       0.36917       0.36897       0.36921
absolute apex (m)              1.30492       1.30329       1.30477
lift (m)                       0.16953       0.16839       0.16911
mean/RMS racket vxy (m/s)            -  0.08120/0.09610 0.08174/0.09668
full angular speed (rad/s)     1.34606       1.34569       1.36186
qvel exceedance                     -       0             0
qacc exceedance                     -       0.00687       0.00706
exact KL                            -       0.00614       0.00653
clip fraction                       -       0.15923       0.16919
```

Both trials passed every strict/recoverability/behavior gate, but LR
`1.25e-4` was rejected because all three primary metrics regressed and its
angular speed, qacc, KL and clip fraction were worse. The accepted LR is
`1.0e-4` with three epochs. Both capped checkpoints are evidence only and must
not be resumed. Complete commands, hashes, failure distribution and negative
result are in
`outputs/rl_sim/v28_survival_primary_experiments_20260823/V28_SURVIVAL_PRIMARY_REPORT.md`.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v28_resume_v27.sh`.
Formal training uses 1024 environments, 128 steps, minibatch 16384, LR
`1.0e-4`, three epochs, clip `0.15`, target KL `0.008`, block validation, no
update cap, and the existing GPU ownership, no-preallocation, host-memory and
78 C guards. W&B continues only the explicitly authorized run `9acnp70r`
with `resume=must` and W&B-only offset `450887680`. The first formal
initialization uploaded one V28 update at `2357985280` before the lineage audit
stopped it. The restart deterministically repeats that update at the same W&B
step and continues at `2358116352`; one 131072-step slot after the stopped V27
tail remains intentionally unfilled, while true optimizer time remains
unchanged. V28 remains experimental, simulation-only, and inherits every
real-camera, replacement-ball, formal RMP/PD coverage and deployment blocker.

## Experimental Measured-QVEL RMP Bounded Ball DR V29

Status (2026-08-23): opt-in continuation implemented, CPU contracts pass, and
the revised Stage-18 semantics bridge passed its protected production-shape
GPU0 acceptance screen. Formal continuation must use the evidence and
launcher below. V28 remains immutable. V29 resumes only V28's Stage-17 pass
archive:

```text
checkpoint  17_rmp28_survival_stability_commit.pkl
true step   1994915840
actor       57-D
critic      221-D
SHA-256     901d08e47749e51bd7305b40fd29a594bb0a66fc0009d0caf90a16325c16ec87
```

The source is the policy saved on successful completion of
`rmp28_survival_stability_commit`, before the wide ball-physics transition.
V28 was stopped at Stage-18 update 174. Over its late window, overall hits,
episode length fraction and full rate plateaued near `8.2/0.56/0.50`, while
episodes with at least three contacts still averaged about `14.2` hits. The
failure was therefore early acquisition for part of the randomized
population, not loss of an already acquired stable cycle.

The exact simultaneous V28 transition was:

```text
parameter                    Stage 17                 Stage 18
mass (kg)                    0.0030625--0.0034375     0.0025--0.0040
normalized inertia          0.40--0.54               0.40--0.6667
initial spin XY (rad/s)      -25--25                  -55--55
initial spin Z (rad/s)       -20--20                  -40--40
gravity Z (m/s^2)            -9.83---9.79             -9.88---9.72
```

V28 Stage 19 would additionally widen ball/racket friction from
`0.16--0.28/0.30--0.48` to `0.10--0.38/0.22--0.62`, solref time from
`3--6 ms` to `2--8 ms`, and damping from `0.74--0.94` to `0.62--1.02`.
V29 changes only these ball/contact axes; reward, gates, resets, actor/critic
observations, measured bounded `q_ref`, RMP/PD, other DR and PPO remain V28.

V29 preserves V28 stages 1--17 exactly and expands the course to 34 stages.
Its replacement bridge/commit ladder is:

```text
parameter                    Stage 18 unchanged     Stage 19 bridge     Stage 20--34
mass (kg)                    .0030625--.0034375     .0030--.0037       .0030--.0040
normalized inertia          0.40--0.54             0.45--0.60         0.54--0.6667
initial spin XY (rad/s)      -25--25                -25--25            -25--25
initial spin Z (rad/s)       -20--20                -20--20            -20--20
gravity Z (m/s^2)            -9.83---9.79           -9.83---9.79       -9.83---9.79
ball sliding friction       0.16--0.28             0.16--0.28         0.16--0.28
racket sliding friction     0.30--0.48             0.30--0.48         0.30--0.48
solref time (s)              .0030--.0060           .0030--.0060       .0030--.0060
solref damping               0.74--0.94             0.74--0.94         0.74--0.94
```

Stage 18 changes course semantics but no physical randomization axis. Stage 19
changes only mass/inertia, and Stage 20 commits their final support. The `2/3`
inertia endpoint is the ideal thin hollow-sphere value, while the learned
`0.54` endpoint becomes the final lower bound. This physical argument does not
substitute for measurement. The user-selected 3--4 g mass interval and every
other V29 bound remain candidate training intervals until new-ball mass and
outcome trials pass the repository evidence contract. The first rejected V29
candidate simultaneously widened mass, inertia and spin at Stage 18; its
30/60-update negative trials are retained in the experiment report.

The preregistered acceptance checks use the same Stage-17 source and production
1024 x 128 shape. A candidate must avoid the V28 acquisition collapse while
retaining V28 safety and flight quality: late-window mean hits at least 13,
`conv_len/full >= 0.90/0.82`, conditional hits at least 13, counted cadence
`0.33--0.44 s`, apex/lift `1.28--1.39/0.145--0.220 m`, mean/RMS racket vxy
at most `0.120/0.170 m/s`, angular speed at most `1.40 rad/s`, zero qvel
exceedance and no material qacc/KL regression. A return increase alone does
not pass. Commands, hashes, metrics, failed candidates and system-safety notes
belong in
`outputs/rl_sim/v29_bounded_ball_dr_experiments_20260823/V29_BOUNDED_BALL_DR_REPORT.md`.

The revised no-physical-change Stage-18 screen used seed `20260822` and the
production 1024 x 128 rollout/PPO shape. At its intentional 40-update
experimental cap, the final 24-update window passed every preregistered metric:
hits `14.1671`, `conv_len=0.94155`, full rate `0.90933`, conditional hits
`14.7460`, counted cadence `0.36442 s`, absolute apex/lift
`1.29317/0.16345 m`, mean/RMS hit-racket vxy `0.08999/0.10480 m/s`, and
angular speed `1.25392 rad/s`. Qvel exceedance was zero; last-12 qacc
exceedance and exact KL averaged `0.00590/0.00639`. The cap is below the
inherited 160-update stage minimum, so this accepts the bridge for uncapped
formal continuation but is not a curriculum graduation. Formal training must
still pass its rolling hold and frozen block validation before Stage 19.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v29_resume_v28.sh`.
It uses 1024 environments, 128 steps, minibatch 16384, LR `1.0e-4`, three
epochs, clip `0.15`, target KL `0.008`, block validation, no update cap and
the existing GPU ownership, no-preallocation, host-memory and 78 C guards.
W&B continues only the explicitly authorized run `9acnp70r` with
`resume=must` and W&B-only offset `473694208`. The last V28 row was true step
`2017722368`, published at `2468610048`; V29's first completed rollout maps to
`2468741120` while the selected checkpoint's true step stays unchanged. V29
remains experimental, simulation-only, and inherits every real-camera,
replacement-ball, formal RMP/PD coverage and deployment blocker.

## Provisional Measured-QVEL RMP Execution-DR Bridges V30

Status (2026-08-23): provisional and evidence-blocked. V29 remains immutable.
The current builder/launcher must not be used for formal continuation until a
matched five-arm frozen-policy motion-distribution audit chooses the execution
range, the builder is reconciled to that result, and the course/reward/PPO
candidate passes a bounded experiment.
V30 resumes only the clean V29 Stage-21 pass archive, before the confounded
Stage-22 execution-DR/reward/target-hold transition:

```text
checkpoint  21_rmp29_launch12_contact_dynamics_wide.pkl
true step   2087190528
actor       57-D
critic      221-D
SHA-256     8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41
```

The source Stage-21 last-24 window had hits `13.6572`, length fraction
`0.91238`, full rate `0.87334`, conditional hits `14.6319`, absolute
apex/lift `1.292996/0.16634 m`, mean hit-racket vxy `0.09562 m/s`, full
racket angular speed `1.21790 rad/s`, exact KL `0.00619`, zero qvel
exceedance and qacc exceedance `0.00490`.

The effective-data audit is fixed by
`rmp_effective_juggling_validity_v3.json`, SHA-256
`391bd03a67ad47e923c8b178c4939d74201169556d6445e6ef6d108f429b0f4f`.
It retains only fresh, physically usable, safety-clean effective juggling at
true ball z `0.90--1.45 m`; it excludes failed/post-drop/stale tails. The
population remains 56 accepted recordings from 64 parseable record_new3/4/5
recordings, with 20,047 post-warmup wall-grid samples. The audit objective is
pointwise joint q/dq/qdd and racket-centre position/orientation/linear and
angular velocity/acceleration coverage, followed by frozen-policy and bounded
PPO safety. It is not a global-min/max fit.

The old broad V2 execution interval is rejected as a default training target.
Even with RMP/PD multipliers `0.5--1.5`, its previous worst-four pointwise
joint q/dq/qdd coverage was only about `90.1/92.6/84.9%`; all 105 candidates
were stable, so interval width was not the remaining nominal closed-loop
calibration solution. The provisional builder tests a staged candidate at
12.5% of the distance from V29 Stage-21 mild DR toward the old broad DR:

```text
parameter                         V29 Stage 21 mild        V30 selected
RMP Kp/Kd multiplier              0.90--1.10               0.85--1.15
estimator process multiplier      0.80--1.25               0.73125--1.40625
estimator measure multiplier      0.75--1.33               0.6575--1.53875
velocity feedforward              0.45--0.55               0.41875--0.58125
acceleration-weight multiplier    0.80--1.25               0.73125--1.46875
target-filter length              9--11                    8--12
RMP output-delay offset (ticks)   -1..1                    -2..2
PD Kp/Kv multiplier               0.90--1.10               0.85--1.15
damping multiplier                0.90--1.10               0.8375--1.1875
armature multiplier               0.90--1.10               0.8375--1.2125
```

This 12.5% interval is not selected or final before the matched audit. The
provisional builder preserves V29 stages 1--21 exactly and has 40 stages. Its
current lessons are:

```text
22  rmp30_rmp_internal_micro_6p25       RMP-internal axes only
23  rmp30_rmp_internal_commit_12p5      RMP-internal selected bound
24  rmp30_pd_plant_micro_6p25           add PD/plant micro bridge
25  rmp30_execution_commit_12p5         full selected execution bound
26  rmp30_target_hold_bridge_3p5        target hold 3.5%, no long tail
27  rmp30_target_hold_commit_5p0        target hold 5% + 1.1% 2--9-tick tail
28--40                                    V29 Stage 23--34 domains in order
```

Execution DR stays at the selected bound through the later course; no later
stage restores V29's 25% or V2's broad intervals. The late 8%/2.2% measured
target-hold setting receives an additional 6.5%/1.65% timing bridge. Later
auxiliary motion, view-centring, contact-centre and path/area/cycle losses are
capped at the Stage-21 strength and their soft limits are not tightened below
Stage 21, so survival remains primary while the original two-sided apex,
lift, cadence, view and safety gates remain health constraints. Every stage
has no update cap.

The stopped V29 Stage-22 last-24 window showed healthy PPO numerics despite
behavior collapse: exact KL `0.00608`, clip `0.1565`, explained variance
`0.8968`, zero raw-action clipping, zero qvel exceedance and qacc exceedance
`0.00550`. V30 therefore keeps LR `1.0e-4`, three epochs, clip `0.15`, target
KL `0.008`, entropy `2e-4`, 1024 environments and 128 steps. The complete
preregistration, range JSONs, trial results and rejected alternatives are in
`outputs/rl_sim/rmp_execution_dr_reaudit_20260823/V30_EXECUTION_DR_REPORT.md`.

The production-shape entry trial used seed `20260822`, 1024 environments and
an intentional 30-update cap at Stage 22. Its final rolling window passed:
hits `13.4798`, `conv_len/full=0.90592/0.85433`, conditional hits `14.3494`,
counted cadence `0.36745 s`, absolute apex/lift `1.29505/0.16689 m`,
mean/RMS hit-racket vxy `0.10045/0.11696 m/s`, and full racket angular speed
`1.23265 rad/s`. Last-12 qvel exceedance was zero, qacc exceedance `0.00526`,
exact protected KL `0.00566`, clip `0.14623`, and explained variance
`0.89866`. Updates 29--30 proposed KL just above `0.008` and were safely
rolled back, so LR is not increased. The cap is below the formal Stage-22
minimum and the trial checkpoint is evidence only; do not resume it.

Evidence-blocked launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v30_resume_v29.sh`.
It uses the existing GPU ownership, no-preallocation, host-memory and 78 C
guards, block validation and no update cap. W&B continues only the explicitly
authorized run `9acnp70r` with `resume=must` and W&B-only offset `561512448`.
V29 ended at true step `2175008768`, published at `2648702976`; V30's first
completed rollout maps to `2648834048` while the source checkpoint's true
optimizer step remains unchanged. V30 remains experimental, simulation-only,
and inherits every real-camera, replacement-ball, formal RMP/PD coverage and
deployment blocker.

## Measured-QVEL RMP Execution-DR Polish V31

Status (2026-08-23): current opt-in experimental continuation. V30 is
preserved unchanged. The user accepted the effective real-trajectory replay
screen as sufficient to proceed with a bounded training experiment; this is
not a formal claim that the candidate covers hardware pointwise. Any later
coverage report must keep only complete intervals between adjacent successful
contacts and exclude the entire failure tail after the last successful
contact.

V31 resumes the same clean V29 Stage-21 pass archive used by V30:

```text
checkpoint  21_rmp29_launch12_contact_dynamics_wide.pkl
true step   2087190528
actor       57-D
critic      221-D
SHA-256     8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41
```

Actor, critic and Adam moments are restored; curriculum history is not. The
first new stage is `rmp31_rmp_internal_micro_6p25`. V31 changes no V30 DR
range, reward, gate, reset, control, observation or PPO value. Its 47-stage
schedule preserves V29 stages 1--21 and then uses:

```text
22  rmp31_rmp_internal_micro_6p25
23  rmp31_rmp_internal_micro_6p25_polish
24  rmp31_rmp_internal_commit_12p5
25  rmp31_rmp_internal_commit_12p5_polish
26  rmp31_pd_plant_micro_6p25
27  rmp31_pd_plant_micro_6p25_polish
28  rmp31_execution_commit_12p5
29  rmp31_execution_commit_12p5_polish
30  rmp31_target_hold_bridge_3p5
31  rmp31_target_hold_bridge_3p5_polish
32  rmp31_target_hold_commit_5p0
33  rmp31_target_hold_commit_5p0_polish
34--47  every remaining V30 domain in order
```

The late 6.5%/1.65% missed-publication bridge also receives an immediate
unchanged-domain polish. The existing final consolidation is retained after
the measured 8%/2.2% setting. Each polish copies the preceding environment,
reward and all gate fields exactly, runs at least 80 updates (140 for the late
bridge), requires at least a 16-update rolling hold and frozen block
validation, and has no update cap. It resets only convergence history; it
does not reset actor, critic or optimizer.

The main optimization contract remains stable `conv_len` and full-episode
rate, with positive hit-event credit capped at 14 and an expected healthy
13--15 hits per 1200-step episode. Apex/lift, true adjacent-contact cadence,
racket mean/RMS vxy, full racket angular speed, view and qvel/qacc are bounded
health metrics. No polish reward is added, so a stage cannot appear improved
while survival falls. The accepted V30 1024-env entry trial supports the
first V31 lesson unchanged: hits `13.4798`, `conv_len/full=0.90592/0.85433`,
cadence `0.36745 s`, apex/lift `1.29505/0.16689 m`,
mean/RMS hit-racket vxy `0.10045/0.11696 m/s`, angular speed `1.23265 rad/s`,
zero qvel exceedance, qacc exceedance `0.00526`, protected KL `0.00566` and
explained variance `0.89866` over the reported windows.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v31_resume_v29.sh`.
It uses 1024 x 128 rollouts, minibatch 16384, LR `1.0e-4`, three epochs, clip
`0.15`, target KL `0.008`, entropy `2e-4`, block validation, no update cap,
no XLA preallocation, the host-memory guard and 78 C stop. LR is intentionally
not raised because the V30 entry trial already exercised protected KL
rollback at updates 29--30.

The user explicitly authorized W&B run `9acnp70r`. V30 did not append formal
points, so V31 uses `resume=must` and W&B-only offset `561512448`; the first
completed V31 rollout maps to `2648834048` while the source checkpoint and
optimizer step remain unchanged. V31 remains experimental, simulation-only,
and inherits every unresolved real-camera, replacement-ball, formal RMP/PD
coverage and deployment-adapter blocker.

## Structured-RMP and RMP-Privileged Critic V35

Status (2026-08-24): preserved bounded diagnostic and RMP-only critic
baseline. V31 remains
immutable, and the V32--V34 runs are bounded diagnostic evidence rather than
formal continuation checkpoints.

V35 resumes the clean passed V31 Stage-23 checkpoint:

```text
checkpoint  23_rmp31_rmp_internal_micro_6p25_polish.pkl
true step   2111569920
actor       57-D
critic      221-D at source; append-only migration to 277-D
SHA-256     93a4002fe579dd34b1768b058c7e81f9c9def937afbfea64cac182a5b98498dc
```

It starts fresh convergence history at Stage 24
`rmp35_rmp_internal_commit_12p5`. It restores the actor, complete old critic,
Adam first/second moments and Adam step. Only 56 new rows in the critic's
first layer and their Adam moments are zero-appended. Actor input and every
actor parameter remain unchanged. The launcher requires
`--allow-obs-dim-migration` and rejects curriculum-history resume,
optimizer/critic reset, another source profile/step/hash, or a mass interval
other than `0.0030--0.0040 kg`.

The diagnosis found two execution-DR correlation errors and one critic-state
omission:

```text
quantity                              V31 sampling         V35 sampling/critic
estimator process/measure             independent x7      one draw, broadcast x7; critic x7
velocity feedforward                  independent x7      one draw, broadcast x7; critic x7
acceleration weight                   independent x7      one draw, broadcast x7; critic x7
target-filter length                  independent x7      one draw, broadcast x7; critic x7
RMP Kp/Kd multiplier                  independent x7      one global draw over calibrated vector; critic x7
XML-PD Kp/Kv multiplier               independent x7      one global draw over calibrated vector; already privileged
RMP output-delay offset               independent x7      unchanged per joint; critic x7
```

Physical RMP configuration files express the first four fields and the extra
gain layers as scalar/global values; fixed per-joint calibration vectors are
still applied. V35 therefore removes the unphysical Cartesian product without
narrowing any interval endpoint. Output-delay uncertainty remains per joint.
Old profiles retain their historical sampling defaults.

The 221-D asymmetric critic was not completely execution-conditioned. It had
true ball/racket state, command errors/history, XML-PD Kp/Kv, damping,
armature, ball/contact DR and legacy timing fields, but none of the randomized
RMP values named above. `critic_sim2real_privileged` did not add them. V35
appends eight normalized seven-joint RMP vectors after the unchanged 221-D
prefix. This changes only value estimation; the 57-D deployment observation
and measured bounded-`q_ref` control path are untouched.

V32's 9.375% bridge plus identical-domain polish did not solve the direct
12.5% drop: its direct-commit final window was hits/length/full
`13.015/0.878/0.815`. Adding success-focused weight `1.25` worsened it to
`12.964/0.872/0.805`, so neither change is promoted. The protected GPU0
screens then measured:

```text
candidate                      hits      conv_len   full      vxy mean/RMS
V31 original first 64          12.8789   0.8705     0.7979    0.1111/0.1291
V33 physical scalar RMP        13.1007   0.8851     0.8191    0.1034/0.1209
V34 structured global gains    13.0381   0.8821     0.8216    0.0975/0.1143
V35 critic, LR 1.0e-4          13.2359   0.8922     0.8297    0.1012/0.1187
V35 critic, LR 1.5e-4          13.0633   0.8856     0.8131    0.1009/0.1183
```

The accepted LR `1.0e-4` screen retained apex/lift
`1.2953/0.1669 m`, adjacent counted cadence `0.3674 s`, full racket angular
speed `1.2089 rad/s`, zero qvel exceedance, qacc exceedance `0.00537`, and
exact KL `0.00571`. Hits, mean length and full-rate slopes were positive over
the final window. It is an effective improvement over the original Stage-24
entry but not a formal graduation because capped `conv_len=0.8922` remains
below `0.90`. LR `1.5e-4` is rejected because its final-window hits, length
and full-rate slopes were all negative.

V35 retains the complete 47-stage V31 schedule, starts at Stage 24, and adds
no curriculum stage. All range endpoints, rewards, gates, resets, control,
actor observations and PPO fields remain unchanged. Formal training uses LR
`1.0e-4`, 1024 x 128 rollouts, three epochs, clip `0.15`, target KL `0.008`,
block validation, no update cap, no XLA preallocation, the host-memory guard
and 78 C stop.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v35_resume_v31_stage23.sh`.
The explicitly authorized W&B run remains `9acnp70r` with `resume=must` and a
W&B-only offset of `678690816`. V31's final true/W&B steps were
`2228748288/2790260736`; V35's first completed rollout is mapped to
`2790391808` while the selected checkpoint and optimizer step remain
unchanged. Full commands, accepted and rejected results, and the interrupted
CPU fallback are recorded in
`outputs/rl_sim/v35_rmp_privileged_critic_experiments_20260824/V35_RMP_PRIVILEGED_CRITIC_REPORT.md`.

V35 remains experimental, simulation-only, and inherits every unresolved
real-camera, replacement-ball, formal RMP/PD coverage and deployment-adapter
blocker.

The attempted formal V35 start on 2026-08-24 was stopped after CUDA and
environment initialization but before the first completed `update=` record.
It did not change the actor/critic/optimizer or append a W&B metric point. Its
output directory is aborted evidence and must not be reused.

## Complete-DR Privileged Critic V36

Status (2026-08-24): preserved; stopped safely on GPU0.

V36 answers the stricter requirement that every realized DR value enter the
asymmetric critic. It resumes exactly the same clean V31 Stage-23 checkpoint,
step, SHA-256, actor and Adam state as V35. It starts fresh at Stage 24
`rmp36_rmp_internal_commit_12p5`, requires append-only observation migration,
and rejects curriculum-history resume, optimizer/critic reset, actor
migration, another source, or a ball-mass declaration other than
`0.0030--0.0040 kg`.

The critic layout is:

```text
221  historical asymmetric critic prefix
 56  V35 recovered-RMP realized parameter block
 91  remaining DR/event/reset randomization complement
---
368  V36 critic input
```

The 91-D complement contains episode delay base/bin, four seven-joint
second-order actuator residual vectors, ball normalized inertia and reset
spin, hard-tail selection, ball-observation position/rotation/velocity bias
and scale, previous-action scale, RMP target-hold and execution-hold state,
missing/dropout/correlated-noise state, randomized observation high-z limit,
and hidden reset/target/episode-limit context. Actor input stays exactly
57-D. The old 221 rows, actor, Adam first/second moments and Adam step are
preserved; all 147 new critic rows and moments start at zero.

`DOMAIN_RANDOMIZED_STATE_FIELDS` and `RESET_RANDOMIZED_STATE_FIELDS` form the
independent sampled-state registry. The legacy, RMP and complete-extra critic
registries must have an exact union with it, and every `EnvState.dr_*` field
must be registered. This regression contract prevents future DR additions
from silently remaining hidden. Stage range endpoints are not observations;
the realized draw is. Independent current observation noise is derivable from
the actor-observation prefix plus true state, while correlated noise state and
dropout/hold counters are explicit.

The matched 64-update GPU0 comparison used the V35 source/seed and identical
1024 x 128 PPO settings. Final 24-update windows were:

```text
candidate                 hits      conv_len   full      full angular
V35 RMP-only critic       13.2359   0.8922     0.8297    1.2089 rad/s
V36 complete DR critic    13.2235   0.8916     0.8292    1.1957 rad/s
```

The primary differences are materially neutral. V36 retained cadence
`0.3669 s`, apex/lift `1.2947/0.1664 m`, recurrent racket vxy mean/RMS
`0.1030/0.1181 m/s`, zero qvel exceedance and qacc exceedance `0.00536`.
Exact KL was `0.00630`, explained variance `0.90283`, and value loss `64.09`.
Hits, mean length and full-rate final-window slopes were positive. Thus the
complete conditioning does not claim a large 64-update performance gain, but
it satisfies the requested state contract without a protected regression.

V36 changes no DR range/probability, stage, reward, gate, reset distribution,
control, actor input or PPO value. It retains all 47 V31 stages, starts at
Stage 24, uses LR `1.0e-4`, 1024 environments x 128 steps, three epochs, clip
`0.15`, target KL `0.008`, block validation and no update cap.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v36_resume_v31_stage23.sh`.
The user-authorized W&B run remains `9acnp70r`, `resume=must`, with W&B-only
offset `678690816`. Because V35 wrote no completed point, V36's first completed
rollout maps to `2790391808`; true checkpoint and optimizer time remain
unchanged. Full evidence is in
`outputs/rl_sim/v36_complete_dr_critic_experiments_20260824/V36_COMPLETE_DR_CRITIC_REPORT.md`.

V36 remains experimental, simulation-only, and inherits every unresolved
real-camera, replacement-ball, formal RMP/PD coverage and deployment-adapter
blocker.

The formal V36 process was stopped safely at Stage 26
`rmp36_pd_plant_micro_6p25`, update 725, true step `2276458496`. The immutable
stop evidence is `mjx_curriculum_last.pkl` in the V36 formal output directory,
SHA-256
`79e8b64cc033d7a2c4fcc4fc8abba51f593fb7e06dfa8c6a37a2b655a386a3c3`.

## Non-Execution-First Full-Episode Course V37

Status (2026-08-24): current opt-in GPU0 continuation; implementation and
contracts complete, formal PPO not started.

V37 implements a curriculum-order correction requested after the V36 stop.
V36 introduced broad recovered-RMP and PD/plant uncertainty before the later
observation/task-support tail, while its length gates remained around
`0.90--0.93`. V37 preserves the complete V36 actor, critic, optimizer,
control, rewards and DR endpoints, but first establishes a protected runnable
policy under all non-execution difficulty.

It resumes only the screened V36 `mjx_curriculum_best.pkl`: source profile
V36, Stage 26, stage update 303, global update 836, true step `2221146112`,
SHA-256
`eb6cf8c1c71c5438aa3ef0eb60a1965f7636bc1990b08a6020f27b2f10a8520d`,
57-D actor, 368-D complete-DR critic and measured ball mass
`0.0030--0.0040 kg`. Curriculum history starts fresh; Adam moments and true
step remain intact. The validator and launcher reject another source, SHA,
profile, stage, step, observation dimension, curriculum-history resume,
optimizer/critic reset, or append-only observation migration.

The preregistered same-domain deterministic screen compared update 250,
automatic best, update 500, update 650 and last over 128 episodes. The selected
best achieved hits/full/length/view `14.180/0.906/1115.9/0.967`, versus last
`13.375/0.859/1088.5/0.950`. Evidence is recorded under
`outputs/rl_sim/v37_source_checkpoint_screen_20260824/`.

The 28-stage order is:

```text
1        recover survival in the learned ball/contact domain
2--19    V36 target-hold, racket geometry, observation calibration,
         camera missing, in-view ball/target XY and final non-execution tail
20       complete non-execution exact-full-episode polish
21--22   V36 RMP-internal 6.25% range + unchanged-domain polish
23--24   V36 RMP-internal complete 12.5% range + polish
25--26   V36 PD/plant 6.25% range + polish
27--28   V36 complete RMP/PD/plant range + polish
```

Stages 1--20 hold these twelve execution intervals at the V36 Stage-21 small
envelope:

```text
RMP Kp/Kd                 0.90--1.10
process variance          0.80--1.25
measurement variance      0.75--1.33
velocity feedforward      0.45--0.55
acceleration weight       0.80--1.25
target filter             9--11 ticks
output-delay offset       -1--+1 ticks
PD Kp/Kv                  0.90--1.10
damping/armature          0.90--1.10
```

The physical correlation contract is final from Stage 1: scalar RMP fields,
RMP gain multipliers and PD multipliers are global episode draws on top of
their calibrated per-joint vectors; only RMP output-delay offsets remain per
joint. The 368-D complete-DR critic is also active from Stage 1. Therefore the
post-polish lessons alter interval support only, not sampling semantics or
critic layout.

Stage 20 is named
`rmp37_complete_nonexecution_full_episode_polish`. It uses V36's final
non-execution configuration, including 8 mm/0.08 m/s in-view observation
noise, widest ball-state/target XY support, camera/missing mechanisms and
8%/2.2% target-publication hold with the 2--9-tick tail, while execution DR
remains small. Its hard gates are rolling `conv_len=1.0` and full-episode rate
`1.0`, held for 24 eligible updates and followed by block validation. There is
no update cap. On graduation the trainer saves
`20_rmp37_complete_nonexecution_full_episode_polish.pkl`; this is the
preregistered runnable checkpoint before broad execution DR.

Stages 21--28 hold every non-execution field exactly fixed at the Stage-20
domain. They restore the exact V36 range ladder, ending at:

```text
RMP Kp/Kd                 0.85--1.15
process variance          0.73125--1.40625
measurement variance      0.6575--1.53875
velocity feedforward      0.41875--0.58125
acceleration weight       0.73125--1.46875
target filter             8--12 ticks
output-delay offset       -2--+2 ticks
PD Kp/Kv                  0.85--1.15
damping                   0.8375--1.1875
armature                  0.8375--1.2125
```

The final V37 environment configuration must equal V36's final environment
configuration exactly. Increment lessons require length/full `0.92/0.84`;
their unchanged-domain polishes require `0.95/0.90`. These later gates may
adapt to broad execution uncertainty without invalidating or overwriting the
protected exact-full Stage-20 checkpoint.

Canonical manual launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v37_resume_v36.sh`.
It uses LR `1.0e-4`, 1024 environments x 128 steps, three epochs, clip `0.15`,
target KL `0.008`, block validation, no stage update cap, no XLA preallocation,
the host-memory guard and 78 C stop. The user explicitly authorized continuing
V36 W&B run `9acnp70r` with `resume=must`. V36's last true/W&B history point is
`2276458496/2955149312`; W&B-only offset `734003200` maps V37's first rollout
from the older selected checkpoint to `2955280384` while true checkpoint and
optimizer time remain unchanged. Repository changes and launcher preflight do
not start training without `CONFIRM_GPU0_READY=YES` and
`ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE=YES`.

V37 remains experimental, simulation-only and non-deployable. Exact-full
simulation survival is not hardware coverage. All unresolved real-camera,
replacement-ball outcome and formal pointwise RMP/PD feedback-coverage
blockers remain.

## Reward-Component-Attributed Survival Course V43

V38--V42 are immutable negative/diagnostic continuations. Their raw final-24
evidence showed return/apex improvement could coexist with primary survival
regression. The decisive frozen screen compared the first episode of the same
256 lanes using V39 best95 and V42 update60 under one V42 environment and seed:
return delta `+1.2327`, full delta `-0.003906`, absolute-apex delta
`+0.00327 m`. Exact additive decomposition attributed `+2.1444` to the
low-apex penalty and `-0.9117` to every other component combined. Evidence and
hashes are registered in
`outputs/rl_sim/v38_gate_reward_experiments_20260824/REWARD_COMPONENT_ROOT_CAUSE_ANALYSIS.md`.

The next course identity is
`goal_d455_measured_qvel_rmp_vertical_v43`. It resumes only V39 Stage-1 best
update 95 at true step `2253586432`, SHA-256
`355f518458e8913e1898c4c17e751984425139ac03f7e484fe64da0be04e8a7c`,
preserving the 57-D actor, 368-D critic and Adam moments while starting fresh
V43 stage/convergence history. The source gate, hash and no-migration/no-reset
contract fail closed.

V43 copies all 28 V42 stages and changes only:

```text
low_hit_penalty_weight                 7000 -> 2500 m^-2
termination_miss_penalty_base         11.46 -> 40
termination_miss_penalty_requires_hit true -> false
full_episode_completion_reward        0 -> 100
```

The miss per-hit cost remains 2.0. Racket miss costs, dense survival, event
quality, apex target/lift/cadence/view rewards, every gate, RMP/PD, bounded
reference, observations, physics, resets and all small/large DR intervals are
unchanged. Frozen counterfactual scoring changes V42-minus-V39 return from
`+1.233` to `-0.648` and the full/failed gap from about 190 to 316.

The preregistered bounded trial used seed `20260901`, 512 environments x 128
steps, minibatch 8192, three epochs, LR `1e-4`, clip `0.15`, target KL `0.011`,
one stage and 128 updates, offline. It failed the raw final-24 Stage-1
length/full gates: `0.92910/0.90034` against `0.95/0.92`; hits were healthy at
`14.1851`. Best 24-update windows reached only `0.93272/0.90306`. The direct
completion reward contributed `0.08170` per step versus total `0.22407`, and
63/64 late updates were accepted, with explained variance about `0.812` and
value loss about `223`. Thus the negative result is not missing reward scale
or pervasive KL rollback. Frozen post-trial paired replay changed full only
from `0.930` to `0.934`, below statistical resolution. V43 is rejected.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v43_resume_v39.sh`.
Do not launch its formal mode or resume either bounded checkpoint. V43 remains
preserved negative evidence, experimental, simulation-only and non-deployable.

## Historical V28 Reward-Bridge Continuation V44

Profile `goal_d455_measured_qvel_rmp_vertical_v44` resumes only the immutable
V39 Stage-1 best update-95 checkpoint at true step `2253586432`, SHA-256
`355f518458e8913e1898c4c17e751984425139ac03f7e484fe64da0be04e8a7c`.
The 57-D actor, 368-D critic, Adam moments and true optimizer time are
preserved; curriculum history starts fresh at
`rmp44_v28_survival_reward_bridge`.

Historical comparison corrected the rollout hypothesis. V28's accepted
formal-shape trial used the same `n_steps=128`, gamma `0.9995`, GAE lambda
`0.99`, LR `1e-4`, clip `0.15`, target KL `0.008` and three epochs, improving
the parent from `conv_len/full=0.94310/0.90665` to `0.95201/0.92818` in 30
updates. Rollout 128 is 0.64 s at the 200 Hz policy rate and spans roughly
1.6--1.8 measured hit cycles; it is not changed in V44.

V37 Stage 1 already uses the exact V28 contact-wide reward values for its
domain, but it skipped the earlier successful V28 survival bridge. V44 inserts
one lesson before the otherwise complete V37 course. Its environment remains
the V37 Stage-1 domain with the same small execution DR and complete critic,
while exactly fourteen secondary weights take their V28 Stage-16 values:

```text
ball-view XY centring
racket-z soft and termination costs
racket-up drift, stability-angular and flatness costs
racket-anchor and ball-miss termination bases
contact-centre excess
cycle path/area/vxy costs
approach racket-vxy and apex-view-centre costs
```

These values are about 15--20% below V37 Stage 1. Primary alive/geometric
survival, apex/lift targets and barriers, hit credit, observations, control,
physics, resets and every DR interval are unchanged. The bridge gates are the
historically successful V28 `conv_len/full=0.93/0.88`; Stage 2
`rmp44_v37_reward_commit` restores V37 Stage-1 weights and its `0.95/0.92`
gates before any new non-execution lesson. The remaining 27 V37 lessons stay
in order, including exact-full non-execution polish before wide RMP/PD DR.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v44_resume_v39.sh`.
The bounded trial used 1024 environments for 30 updates and rejected V44.
Raw final-12 length/full/hits were `0.92396/0.88843/14.2112`; full sloped
downward by `0.000173/update`. Frozen 256-lane paired replay under the same
V44 environment measured V39-source versus V44-last full
`0.91797 -> 0.88281` and mean length `1127.3 -> 1105.2`. The transition table
was seven failed-to-full improvements against sixteen full-to-failed
regressions. Failed episodes survived longer and gained more hits, but that
local improvement traded away more already successful DR lanes.

Additive reward comparison does not show a scalar component defeating the
primary objective. Dense survival, post-first-hit alive, hit-count-floor,
hit, apex/lift and total credit all fell with full. Small improvements in
racket angular, cycle-path and cycle-vxy components were far too small to
offset those losses. Thus the root cause is missing protection of the source
policy's successful state distribution under a mean PPO objective. Do not
launch V44 formally or resume its checkpoint. It remains preserved negative
evidence.

## Successful-State Replay-Anchor Polish V45

Profile `goal_d455_measured_qvel_rmp_vertical_v45` resumes only the immutable
V39 Stage-1 best update-95 checkpoint at true step `2253586432`, SHA-256
`355f518458e8913e1898c4c17e751984425139ac03f7e484fe64da0be04e8a7c`.
It preserves the 57-D actor, 368-D critic and Adam moments, but starts fresh
history at `rmp45_successful_replay_survival_polish`. V45's 28 stages are
value-for-value V39 copies: reward, every gate, observation/control layout,
RMP/PD, bounded reference, resets, physics, small pre-polish execution DR and
late complete V36 execution DR are unchanged. Only stage identity/notes and
launcher-side PPO preservation controls differ.

The replay set was collected from the frozen V39 source over 256 first-episode
DR lanes at seed `20260905`. Exactly 227 lanes reached the 1200-step time
limit and 29 failed. Step-major reconstruction retained only the 272,400
observations belonging to those 227 successful lanes, then deterministically
subsampled them to `65536 x 57` float32. Canonical artifact:
`outputs/rl_sim/v38_gate_reward_experiments_20260824/v39_source_replay_seed20260905_full_only_65536.npy`,
SHA-256 `850b581f23d63d7eea012414b70d73bda412d913c35024a202f5facbbdfe2042`.
The adjacent JSON hashes the source checkpoint, episode CSV, unfiltered
observation stream and filtered output. Failed-lane observations are not used
as preservation targets.

V45 copies the earlier successful GPU0 replay-anchor polish optimizer rather
than changing one more reward component: 1024 environments, rollout 128,
gamma `0.9995`, GAE lambda `0.99`, two epochs, LR `5e-6`, clip `0.08`, target
KL `0.003`, entropy zero, current-source KL coefficient `0.02`, replay KL
coefficient `0.05`, no hard replay-KL invention, and failure focus disabled.
The source policy is passed explicitly as the KL reference. The current
`-4.8` log-standard-deviation floor is retained; the historical `-3.6` floor
is not copied because it would abruptly increase exploration in this source.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v45_resume_v39.sh`.
The unique bounded run
`outputs/rl_sim/v45_successful_replay_trial_seed20260906_u64` used 1024
environments for 64 updates and rejected V45. Its best 12-update raw window
was length/full/hits `0.94051/0.91078/14.4490`; final-24 was
`0.92322/0.88997/14.1565`. The final-24 absolute apex/lift were
`1.28486/0.15775 m`, view was `0.96973`, qvel exceedance was zero and qacc
exceedance was `0.00463`, so secondary/safety regression was not the blocker.
Critic EV remained about `0.963`, per-update exact KL about `0.00087`, and no
update rolled back. However replay KL from the V39 success policy accumulated
to `0.00848`; the best early window occupied roughly `0.004--0.006`. V45
therefore confirms that soft anchoring alone does not change the direction of
the transition-mean PPO gradient or bound its cumulative drift. Do not launch
V45 formally or resume its checkpoints.

## Preserved Full-Success Lane-Risk Diagnostic V46

Profile `goal_d455_measured_qvel_rmp_vertical_v46` resumes the same immutable
V39 Stage-1 best update-95 checkpoint and preserves its actor, 368-D critic
and Adam moments. Its 28 environment stages are value-for-value V39 copies;
the first identity is `rmp46_full_success_risk_polish`. All reward components,
gates, RMP/PD, bounded reference, observations, resets, physics, small early
execution DR and final complete V36 execution DR remain unchanged.

V46 fixes the optimizer-level mismatch established by V44/V45. The PPO batch
normally averages transition advantages, so it may improve some failed lanes
while breaking more already-full lanes. For every time-limit-completed lane
with at least 13 hits, V46 weights only its positive normalized advantages by
`3.0` within the causally attributable final 128 rollout transitions.
Negative advantages in those lanes retain unit weight, and failed and
unfinished transitions remain in the ordinary PPO objective. After weighting,
advantages are RMS-renormalized. This directly raises successful-lane actor
credit without adding a terminal reward, changing critic targets or excluding
hard cases.

V46 retains rollout 128, gamma `0.9995`, GAE lambda `0.99`, two epochs, LR
`5e-6`, clip `0.08`, target KL `0.003`, entropy zero, current/replay soft KL
coefficients `0.02/0.05`, and failure focus off. It additionally caps absolute
source KL on the hashed 65536-row successful replay at `0.006`, selected from
V45's observed positive-to-stalled transition. An over-bound candidate is
backtracked between the behavior and candidate policies to the largest safe
fraction; it is not blindly discarded. Focus fraction, candidate/post replay
KL, projection rate/scale and no-safe-step count are mandatory evidence.

Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v46_resume_v39.sh`.
The initial 64-update bounded run's final-12 length/full/hits were
`0.93907/0.91348/14.3741`, but this short positive tail did not replicate. The
independent 128-update run
`outputs/rl_sim/v46_full_success_risk_trial_seed20260907_u128` ended with
final-24 length/full/hits `0.93036/0.90015/14.2341`; raw length and hits sloped
`-0.552 step/update` and `-0.00675 hit/update`. Its best rolling length/full at
update 109 was only `0.93420/0.90518`.

Secondary evidence was stable: final-24 apex/lift were
`1.28572/0.15771 m`, view `0.96835`, mean/RMS hit-racket vxy
`0.09632/0.11114 m/s`, full angular speed `1.17803 rad/s`, qvel exceedance
zero, qacc exceedance `0.00477`, EV `0.9615`, and exact PPO KL `0.00115`.
Replay KL remained below the `0.006` boundary at `0.00561`; projection fired
on one third of final updates without a no-safe-step rollback. Yet reward per
step rose by `+0.00542` from first to final 24, primarily through dense/alive,
lower low-apex loss and fewer workspace terminal costs, while full did not
rise. V46 is therefore rejected: the scalar rewards and preservation boundary
work, but success focus occupies only `0.0161` of samples, so multiplying it by
three adds only about 3.2% actor mass and gives rare failed tails no fixed
share. Do not launch V46 formally or resume its checkpoints.

## Preserved Negative Balanced-Outcome Diagnostic V47

Profile `goal_d455_measured_qvel_rmp_vertical_v47` resumes only the immutable
V39 Stage-1 update-95 checkpoint at true step `2253586432`, SHA-256
`355f518458e8913e1898c4c17e751984425139ac03f7e484fe64da0be04e8a7c`.
It preserves the 57-D actor, 368-D critic and Adam moments but starts a fresh
window at `rmp47_balanced_outcome_risk_polish`. Its complete 28-stage reward,
gate, observation, RMP/PD, bounded-reference, reset, physics and DR contract is
value-for-value V39; only stage identity and launcher-side PPO semantics differ.

V47 retains rollout 128 because the task is locally corrective and the final
0.64 s contains at least one actionable juggling cycle. It corrects the actual
failure of the transition-mean objective: rare completed groups lose aggregate
gradient mass as their population fraction shrinks. For completed true
terminations below 13 hits, only negative normalized advantages in the final
128 attributable steps receive `0.15` additional batch-average actor mass. For
completed time-limit lanes with at least 13 hits, only positive advantages in
the same tail receive another `0.15`. If both groups exist, their fixed masses
raise mean sample weight from `1.0` to `1.30`; RMS normalization then preserves
the established optimizer scale. Empty groups add zero, wrong-sign advantages
retain ordinary weight, and unfinished suffixes never borrow a later outcome.

All preservation controls remain V46 values: 1024 environments, two epochs,
LR `5e-6`, clip `0.08`, target KL `0.003`, entropy zero, current/replay KL
coefficients `0.02/0.05`, the hashed successful replay set, and absolute replay
KL limit `0.006`. Preserved diagnostic launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v47_resume_v39.sh`.

The 128-update bounded run at
`outputs/rl_sim/v47_balanced_outcome_risk_trial_seed20260908_u128` rejects
V47. Short-term oscillation was explicitly allowed: the run fell to a middle
platform then recovered. Nevertheless, episode-weighted length/full were
`0.92524/0.89901` over updates 9--32 and `0.92793/0.89594` over updates
97--128, while the best mature rolling pair was only `0.93059/0.90081`.
This is an approximately flat length platform with lower full completion, not
an oscillatory upward trend. Apex/lift, view, racket motion and qvel/qacc
remained within their existing bounds.

Frozen first-episode evaluation paired all 256 lanes at seed `20260909`.
V39 source, V47 best and V47 last produced full counts `241/235/237`, mean
lengths `1147.73/1130.85/1135.58`, and mean hits
`14.672/14.492/14.516`. Best rescued four failed source lanes but broke ten
successful lanes; last rescued five and broke nine. Artifacts are
`frozen_paired_seed20260909.csv` and
`frozen_paired_episodes_seed20260909.csv` inside the V47 run directory. Do not
launch V47 formally or resume either checkpoint.

The combined V43--V47 evidence identifies an update-acceptance defect, not a
remaining reward coefficient. Scalar PPO reward and mean KL can improve while
the discontinuous episode outcomes exchange successful and failed lanes; the
stage gate only blocks later graduation, and `stage_best_score` remains a
weighted return/hit/length/quality sum rather than a primary-metric commit
rule. No further reward or optimizer parameter variant is authorized as the
next remedy. A successor must first implement a legacy-default-off,
transactional candidate/incumbent comparison on identical frozen lanes. It
may tolerate short-term candidate oscillation, but actor, critic and Adam state
are promoted only when the length/full platform improves without taking any
already-in-range secondary or safety metric out of range; rejected candidate
state is rolled back atomically. Formal W&B `9acnp70r` and tmux `pp_gpu0`
remain untouched until that contract has regression and bounded evidence.

## Preserved Transactional Main-Metric Experiment V48

Profile `goal_d455_measured_qvel_rmp_vertical_v48` is the implemented
successor. Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v48_resume_v39.sh`.
It resumes only V39 Stage-1 best update 95 at true step `2253586432`, SHA-256
`355f518458e8913e1898c4c17e751984425139ac03f7e484fe64da0be04e8a7c`,
preserving the 57-D actor, 368-D critic and Adam moments. V48's 28 stages are
value-for-value V39 copies except for monotonically versioned names and notes.
Reward, every gate, observations, bounded q_ref, RMP/PD, physics, resets, all
small/large DR intervals and PPO values are unchanged. The first stage is
`rmp48_transactional_main_metric_polish`. Resume history, reset/migration,
another source and V46/V47 outcome weighting are prohibited.

The only behavioral change is legacy-default-off transactional promotion:

```text
candidate block            8 PPO updates
paired evaluation          256 deterministic first episodes, seed 20260909
primary below gate         each length/full gap non-increasing; sum strictly decreases
primary inside gate        both remain inside their accepted lower bands
secondary behavior         interval gap may not increase
arm safety                 qvel exceedance = 0; qacc exceedance <= 0.01
state transition           commit or restore actor + critic + complete Adam state
```

The same reset/DR lane is paired by environment index, so gained and lost full
episodes are reported directly. Candidate rollouts may oscillate within the
eight-update block; only the accepted incumbent series is the stable-policy
history. Global environment-interaction steps continue to count rejected
attempts, while Adam time rolls back. Regular last/best/archive checkpoints
are written only at an accepted boundary. Signal and temperature stops save
the accepted incumbent; a numerical safety stop separately retains the bad
candidate for diagnosis.

One preregistered bounded test, not a parameter sweep, ran at
`outputs/rl_sim/v48_transactional_rootfix_trial_seed20260829_u32` with 1024
environments x 128 steps, minibatch 16384, three epochs, LR `1e-4`, clip
`0.15`, target KL `0.010`, seed `20260829`, one stage and 32 attempted
updates. Runtime trainer/launcher hashes were
`10c576ca12072ed23565181c2244bcf60fa468a60fa9d85c10d0dbcd1cc95b75`
and `d15e9f5dbfa96fd039492ffe554f05e2a9f4feec48e8331dcf2682a63fed7ad6`.
The paired frozen results were:

| boundary | incumbent length/full | candidate length/full | lane +/− | result |
| --- | --- | --- | --- | --- |
| 8 | 0.92379 / 0.89062 | 0.94777 / 0.92578 | +13 / −4 | accept |
| 16 | 0.94777 / 0.92578 | 0.94945 / 0.92188 | +8 / −9 | accept |
| 24 | 0.94945 / 0.92188 | 0.93973 / 0.90234 | +8 / −13 | reject/rollback |
| 32 | 0.94945 / 0.92188 | 0.93717 / 0.89844 | +7 / −13 | reject/rollback |

All four boundaries reported zero secondary/safety gap regressions. Raw
training-rollout qvel exceedance was zero and qacc exceedance was
`0.00449--0.00480`; exact paired safety values are logged by the post-trial
audit extension in subsequent runs. The source/final Adam counters were
`461272/461656`: the delta of 384 equals 16 accepted updates x 3 epochs x 8
minibatches, rather than the 768 steps that 32 unconditionally committed
updates would have produced. This is direct rollback evidence. Final accepted
checkpoint SHA-256 is
`70ced04838db78f6a4083736ae2fbf6f6727f570d9e6c9484312f04d76da0530`;
progress CSV SHA-256 is
`ce5a9cd2cafb950ca77d41d44c65c03b201bce526d8fbc06cabb3a47ae0e77ab`.

The cap produced positive root-fix evidence, not a graduated policy. Frozen
accepted full crossed and retained its `0.92` gate, while length improved
monotonically but stopped at `0.94945`, still `0.00055` below `0.95`; the raw
rolling online window was also not converged. Therefore the bounded
checkpoint must not be presented as the finished model or promoted directly
to hardware. Formal V48 restarts from the immutable V39 source with the same
seed `20260829`, removes the update cap, and continues W&B run `9acnp70r` with
`resume=must` plus W&B-only offset `728039424`. It must run in tmux `pp_gpu0`.
Judge progress by accepted transaction boundaries; raw candidate oscillation
is expected. Do not weaken the 8-update/256-lane contract or scan reward/PPO
parameters after rejection. V48 remains experimental, simulation-only and
inherits all unresolved real-camera, new-ball and formal RMP/PD coverage
blockers.

Formal V48 was started in tmux `pp_gpu0` on 2026-08-24/25 with explicit seed
`20260829` and output directory
`outputs/rl_sim/measured_qvel_rmp_vertical_v48_gpu0_seed20260829_20260824_transactional_resume_v39_best95`.
W&B `9acnp70r` resumed with the registered offset. The process-local paired
baseline was length/full `0.92795/0.89453`. At the first update-8 boundary the
candidate reached `0.94257/0.92188`, with full-lane transitions `+11/-4`, zero
secondary/safety gap regressions, qvel exceedance `0`, and qacc exceedance
`0.004428`; actor, critic and Adam were committed. The user authorized a safe
stop on 2026-08-25 after attempted update 107. The complete boundary record
was:

| boundary | incumbent -> candidate length/full | lane +/− | decision |
| --- | --- | --- | --- |
| 8 | `0.92795/0.89453 -> 0.94257/0.92188` | `+11/-4` | accept |
| 16--72 | incumbent `0.94257/0.92188`; seven candidates | each net negative | reject x7 |
| 80 | `0.94257/0.92188 -> 0.94390/0.92188` | `+5/-5` | accept |
| 88--104 | incumbent `0.94390/0.92188`; three candidates | each net negative | reject x3 |

Thus only two of 13 complete transactions committed. Every boundary retained
zero qvel exceedance, qacc exceedance stayed about `0.00437--0.00462`, and no
secondary gap regression was reported; most rejection came solely from the
hard primary fixed-lane comparison. The final accepted complete-state
checkpoint is update 80, true step `2264072192`, SHA-256
`f7ee8ca8dd48347b8c8eb1c96a7b8a02223d329db5b1038f092f3e5a8ad0be07`.
This proves rollback integrity but rejects transactional promotion as the
long-training optimizer path: difficult DR experience is sampled, yet safe
candidate adaptation can be repeatedly discarded because a temporary
primary regression is not allowed. Preserve V48 and its artifacts as a
diagnostic; do not resume it for the current course.

## Normal Continuous Long-Trend Continuation V49

Profile `goal_d455_measured_qvel_rmp_vertical_v49` is the current successor.
Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v49_resume_v48.sh`.
It resumes only the V48 accepted update-80 checkpoint above, preserving actor,
critic and complete Adam state while starting a fresh convergence window at
`rmp49_normal_long_trend_polish`. The source profile, stage, true step,
dimensions, ball-mass interval and SHA are fail-closed. Curriculum-history
resume, reset/migration, another source, transaction mode and V46/V47 outcome
weighting are prohibited.

V49 changes only update promotion: the frozen-lane transaction guard is off,
so every numerically valid PPO update feeds the next rollout. The 28-stage
environment is otherwise value-for-value V39/V48: identical reward terms,
graduation gates, bounded q_ref and recovered RMP/PD path, observations,
physics, resets, all non-execution/execution DR ranges and small-before-wide
DR ordering. PPO is also unchanged:

```text
n_envs / n_steps       1024 / 128
minibatch / epochs     16384 / 3
learning rate          1e-4
clip / target KL       0.15 / 0.010
entropy                0.0002
gamma / GAE lambda     0.9995 / 0.99
stage update cap       none
trend window           24 committed normal updates
```

Normal training permits short-term `conv_len/full` regression while learning
hard DR cases. This does not disable quality or safety enforcement: the
ordinary strict rolling gates and 128-lane block validation still control
graduation, and qvel/qacc, view, apex, contact and motion-quality metrics
remain logged and gated. Judge the experiment from sustained 24-update and
longer trends, not individual points and not an expectation of monotonic
updates. No reward, DR or PPO scan should be introduced before this long run
has enough history.

V49 continues W&B run `9acnp70r` with `resume=must`. V48's last visible point
was attempted update 107 at true step `2267611136` plus offset `728039424`;
the source checkpoint is accepted update 80 at `2264072192`. The V49 W&B-only
offset is therefore `731578368`, so its first 131072-step rollout follows the
V48 tail while true optimizer/checkpoint time remains unchanged. The formal
launcher has no stage cap and runs only in tmux `pp_gpu0`. V49 remains
experimental, simulation-only and inherits every unresolved real-camera,
replacement-ball and formal pointwise RMP/PD coverage blocker.

## Preserved Strict Long-Block Retention Diagnostic V50

Profile `goal_d455_measured_qvel_rmp_vertical_v50` is a preserved diagnostic.
Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v50_resume_v49.sh`.
It resumes only V49 Stage-1 periodic update 850 at true step `2375483392`,
SHA-256
`1e611b69373c2d1fe636578c7f88030c78f22e2b7775e603fe865581f96ea2ef`,
with the 57-D actor, 368-D complete-DR critic and Adam moments intact. It
starts fresh convergence history at `rmp50_long_block_retention_polish`.
Another source, curriculum-history resume, optimizer/critic reset,
observation migration and outcome weighting fail closed.

V49 supplied the required long-trend failure evidence. It remained in Stage
1 through attempted update 854. Its raw update ranges moved as follows:

| V49 update range | conv_len | full | hits | exact KL | qacc exceed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 385--576 | 0.93269 | 0.89998 | 13.9920 | 0.00700 | 0.00441 |
| 577--768 | 0.92925 | 0.89297 | 13.8748 | 0.00707 | 0.00435 |
| 769--854 | 0.92692 | 0.88661 | 13.7742 | 0.00726 | 0.00436 |

The best raw 24-update length window ended at update 432 with
`conv_len/full=0.94427/0.91815`, still below the unchanged `0.95/0.92`
gates. PPO was numerically healthy and the critic explained variance reached
about `0.993`; unrestricted continuation was therefore rejected as policy
retention failure, not optimizer divergence.

Training-window selection was also misleading. A deterministic paired
diagnostic of V48, V49 archives 350--550, V49 internal best and V49 periodic
last used 256 first-episode lanes at seed `20260910`, but explicitly forced
both observation-missing probabilities to zero. In that easier no-missing
domain, the selected update-850 state measured hits `14.21094`, length
`1143.26/1200 = 0.95271`, full `0.92578` and view `0.96634`. The internal
update-432 best measured only hits `14.15625`, length `1127.24/1200 =
0.93937`, full `0.91406` and view `0.95781`. These results establish latent
no-missing capability and a checkpoint-selection mismatch, but they are not a
complete Stage-1 gate pass. V50's transaction evaluator retains the full
Stage-1 observation-missing and coherent-missing contract; it measured the
update-850 incumbent at length/full `0.92587/0.89844`, which remain below the
unchanged `0.95/0.92` gates. Results and per-lane outcomes from the no-missing
diagnostic are preserved in
`outputs/rl_sim/v50_plateau_rewind_experiments_20260825/` with CSV SHA-256
`7baf394e3a301d540f2f6c5861bc39f928f9f4d5707bfda5abf67e7f7a6f9fdf`
and
`2ef9923ef8ad84cbaaadcf4357c3ccb3169868e810421dbb99df4796866b58c0`.

V50 changes only policy-state promotion. Every candidate receives 128 normal
PPO updates before comparison with the accepted incumbent on the same 256
deterministic first-episode lanes at seed `20260910`. Actor, critic and every
Adam moment commit or roll back atomically. While a primary gate is missed,
the individual length/full gaps may not increase and their combined gap must
decrease. Once both are in range, a candidate must retain both ranges. Every
active secondary gate must remain in range or move no farther away; qvel
exceedance must be zero and qacc exceedance at most `0.01`. The 128-update
interval is 16 times V48's rejected block and must not be shortened. The
ordinary 24-update rolling gates and independent stochastic block validation
remain the only graduation contract.

All 28 stage configurations and PPO values remain value-for-value V49:

```text
n_envs / n_steps             1024 / 128
minibatch / epochs           16384 / 3
learning rate                1e-4
clip / target KL             0.15 / 0.010
entropy                      0.0002
gamma / GAE lambda           0.9995 / 0.99
transaction interval/lanes   128 / 256
transaction seed             20260910
stage update cap             none
rolling window               24 committed updates
```

The unused formal launcher pins W&B `9acnp70r` with `resume=must` and
W&B-only offset `732102656`, but the bounded result below rejects V50 for
formal training. V50 remains experimental, simulation-only and inherits every
unresolved real-camera, replacement-ball and formal pointwise RMP/PD coverage
blocker.

The preregistered production-shape bounded run is
`outputs/rl_sim/v50_long_block_retention_trial_seed20260911_u128`. At its
single update-128 boundary:

| metric | incumbent | candidate | change |
| --- | ---: | ---: | ---: |
| length fraction | 0.925872 | 0.932253 | +0.006380 |
| full rate | 0.898438 | 0.894531 | -0.003906 = -1/256 |
| combined primary gap | 0.045690 | 0.043216 | -0.002474 |
| full lanes | — | — | +13 / -14 |
| secondary regressions | — | 0 | safe |
| qvel / qacc exceedance | — | 0 / 0.004199 | safe |

The candidate was correctly rejected by V50's exact individual
non-regression rule. The saved rollback parameter and optimizer tree digests
are respectively
`25f2e1430691e8a2f60fede665f6e01325f7c2ac5ed9758367ddde10618ebaba`
and
`9d913c33d1577d57b3e9f0fcfb3d1fe0d9ad6611b6d412193dac4f62abc30ea8`,
exactly equal to the V49 source; Adam `t` remains `481901`. The final 24 raw
updates retained mean length `1118.45/1200`, hits `13.8949`, exact KL
`0.00673`, zero qvel exceedance, qacc exceedance `0.00434` and explained
variance `0.99343`. V50 therefore validates atomic rollback but is rejected
for formal training because a one-lane quantization change can veto a lower
combined unchanged-gate gap.

## One-Lane-Resolution Retention Continuation V51

Profile `goal_d455_measured_qvel_rmp_vertical_v51` is the current successor.
Canonical launcher:
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v51_resume_v49.sh`.
It resumes the same V49 Stage-1 update-850 checkpoint at true step
`2375483392`, SHA-256
`1e611b69373c2d1fe636578c7f88030c78f22e2b7775e603fe865581f96ea2ef`,
with actor, 368-D critic and Adam state intact. It starts fresh history at
`rmp51_one_lane_resolution_retention_polish` and rejects another source,
curriculum-history resume, optimizer/critic reset, observation migration and
outcome reweighting.

Every one of V50's 28 stages and every PPO, reward, environment, control,
observation, reset, physics, DR and curriculum-gate value is preserved. V51
changes only the below-gate full-rate comparison in the transaction guard:

```text
transaction lanes                    256
individual length-gap tolerance      0
individual full-gap tolerance        1/256 = 0.00390625
combined primary-gap improvement     strictly positive
in-band 0.95 / 0.92 requirement      both remain in range
secondary gap tolerance              1e-12
qvel / qacc exceedance limits        0 / 0.01
transaction interval                 128 updates
```

The tolerance is measurement resolution, not a curriculum-gate relaxation.
It applies only while an incumbent is below a primary gate and cannot promote
a candidate back below either gate after both pass. The V50 boundary is the
fixed regression case: one full lane is tolerated because length and combined
gap improve and the complete secondary/safety audit passes. Larger primary
trades remain rejected. The committed 24-update rolling gates and independent
stochastic block validation remain the only graduation evidence.

Formal V51 uses the same 1024 x 128, minibatch 16384, three epochs, LR `1e-4`,
clip `0.15`, target KL `0.010`, entropy `0.0002`, gamma `0.9995` and GAE
lambda `0.99` contract. Online continuation of W&B `9acnp70r` with
`resume=must` and offset `732102656` requires explicit V51 external-upload
authorization. Because that approval was not granted, the active 2026-08-25
run uses W&B offline and remains entirely local:
`outputs/rl_sim/measured_qvel_rmp_vertical_v51_gpu0_seed20260911_20260825_one_lane_retention_resume_v49_last850_offline`.

The first production transaction committed at update 128:

| metric | incumbent | candidate | result |
| --- | ---: | ---: | --- |
| length fraction | 0.924017 | 0.928281 | improve |
| full rate | 0.882812 | 0.886719 | improve |
| combined primary gap | 0.063171 | 0.055000 | improve |
| full lanes | — | +15 / -14 | net +1 |
| secondary regressions | — | 0 | safe |
| qvel / qacc exceedance | — | 0 / 0.004187 | safe |

The accepted `mjx_curriculum_best.pkl` is true step `2392260608`, global and
stage update 128, Adam `t=484973`, SHA-256
`c4b7eda6dce751954f9ea55cd2771309bde92df2b39f0114513bb07171f60e5d`.
Training continued beyond update 130 in tmux `pp_gpu0`. This proves the V51
promotion strategy can commit full-domain primary progress without changing
the gates; it is not yet Stage-1 graduation evidence. V51 remains
experimental, simulation-only and inherits every unresolved real-camera,
replacement-ball and formal pointwise RMP/PD coverage blocker.

## Bounded Hard-Lane Gradient-Coverage Trial V52

Profile `goal_d455_measured_qvel_rmp_vertical_v52` is a bounded diagnostic,
launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v52_resume_v51.sh`.
It resumes V51's Stage-1 `mjx_curriculum_best.pkl` at global/stage update 256,
true step `2409037824`, SHA-256
`c6b150781a37648277e7994c2a074f75c40005f8aaaeffcc27ae4eff972853fb`.
The 57-D actor, 368-D critic and Adam moments are restored, but V52 starts a
fresh convergence history at `rmp52_hard_lane_gradient_polish` because actor
gradient allocation changes.

The 3361-update V51 audit found no PPO numerical failure: KL averaged about
`0.00715`, clip fraction `0.183`, explained variance `0.9944`, qvel
exceedance was zero and qacc exceedance averaged `0.00448`. It did find a
stable physical/reset split in an independent 1024-lane screen. Lanes with at
least two of the seven frozen hard-tail indicators achieved deterministic
length/full `0.90093/0.84461`, versus `0.97115/0.95388` for ordinary lanes;
the stochastic split was `0.89900/0.85192` versus `0.96978/0.94340`.

V52 therefore preserves all 28 V51 stages value-for-value, including every
reward coefficient and the unchanged Stage-1 `0.95/0.92` length/full gates.
PPO remains 1024 environments, rollout 128, minibatch 16384, three epochs,
LR `1e-4`, clip `0.15`, target KL `0.010`, entropy `0.0002`, gamma `0.9995`
and GAE lambda `0.99`. Only the actor advantage is reweighted:

| hard condition | threshold |
| --- | ---: |
| ball solref time | `>=0.00525 s` |
| reset ball spin-x | `<=-12.5 rad/s` |
| aggregate reset disturbance | `>=3.20` |
| normalized ball inertia | `>=0.635` |
| episode target-y | `>=0.035 m` |
| reset ball-vxy | `>=0.0029 m/s` |
| ball mass | `>=0.00375 kg` |

At least two conditions must be simultaneously active. Such transitions use
actor-advantage weight `1.5`; all ordinary transitions keep weight `1.0`, and
the critic remains unweighted. No range, reset probability, actor observation
or deployment contract changes. V46/V47 outcome weighting is prohibited.

The offline 64-update trial was rejected. Deterministic hard-lane length/full
improved `+0.00242/+0.01250`, below the preregistered `+0.01/+0.02`, while
ordinary length regressed `-0.00530`, just beyond its protection limit. The
complete stochastic screen regressed length/full `-0.00981/-0.01660`.
Mean approximate/exact KL remained `0.00706/0.00704`, qvel exceedance was
zero, qacc exceedance was `0.00440`, and absolute apex remained `1.29890 m`.
Thus V52 is an objective failure, not a numerical or safety failure. Do not
resume it. Lane-level evidence and the decision report are under
`outputs/rl_sim/v52_multiseed_generalization_screen_20260825/`.

## Bounded Reward-Only Apex/Survival Balance Trial V53

Profile `goal_d455_measured_qvel_rmp_vertical_v53` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v53_resume_v51.sh`.
It restarts from the frozen V51 Stage-1 update-256 best checkpoint, true step
`2409037824`, SHA-256
`c6b150781a37648277e7994c2a074f75c40005f8aaaeffcc27ae4eff972853fb`.
V52 state is prohibited. Actor, 368-D critic and Adam moments are restored;
the convergence history is fresh at `rmp53_apex_survival_balance_polish`.

The V51 3361-update attribution distinguishes two apex signals. The narrow
symmetric 1.30 m target reward is positively aligned with future full-episode
rate and remains at weight `4.0` with sigma `0.035 m`. The one-sided loss
below the exact 1.28 m lower edge becomes less negative while full rate falls,
showing the apex-for-survival trade. V53 therefore changes exactly:

| field | V51 | V53 |
| --- | ---: | ---: |
| `low_hit_penalty_weight` | `7000 m^-2` | `5000 m^-2` |
| `post_hit_survival_reward_weight` | `3.5` | `4.5` |

No completion reward, extra termination cost or advantage reweighting is
added. Every gate, symmetric apex/lift reward, environment, physics, reset,
DR, actor/critic observation, RMP/PD/control, PPO and entropy value remains
V51-exact. The fixed PPO contract is 1024 environments, rollout 128,
minibatch 16384, three epochs, LR `1e-4`, clip `0.15`, target KL `0.010`,
entropy `0.0002`, gamma `0.9995` and GAE lambda `0.99`.

The offline 64-update result failed. Deterministic length/full changed
`0.933643/0.895508 -> 0.930454/0.892578`; stochastic changed
`0.931973/0.894531 -> 0.929492/0.893555`. Mean approximate/exact KL was
`0.00735/0.00733`, explained variance `0.9922`, qvel exceedance zero, qacc
exceedance `0.00445`, and absolute apex `1.29718 m`, so this was not an
optimizer, safety or mean-apex failure. Deterministic `ball_too_low`
terminations increased from 45 to 52. The weakened lower-apex barrier caused
the exact tail it was meant to trade against, proving that the prior negative
historical correlation was confounded by shorter failed episodes. Reject V53,
do not resume it, and prohibit an uncapped run. Any next reward ablation must
restore `low_hit_penalty_weight=7000` and isolate survival credit. Evidence is
under `outputs/rl_sim/v52_multiseed_generalization_screen_20260825/`.

## Old-Course Reentry Before the Complete V37 Course V54

Status (2026-08-25): current opt-in GPU0 continuation; local W&B offline.

Profile `goal_d455_measured_qvel_rmp_vertical_v54` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v54_resume_v36_stage25.sh`.
V54 rejects every V37--V53 checkpoint and resumes only the passed V36
Stage-25 archive `25_rmp36_rmp_internal_commit_12p5_polish.pkl`, true step
`2181431296`, SHA-256
`310fc93f8c69e0f4fcefe59a161948e33cb1b23e57f880d75050dd56713f6a1f`.
It preserves the 57-D actor, 368-D complete-DR critic, Adam moments and true
optimizer time, while starting fresh curriculum history at
`rmp54_v36_stage25_reentry_polish`. Curriculum-history resume, actor/critic
migration, optimizer/critic reset, another source, transaction promotion and
advantage reweighting must fail closed.

The source was selected on the exact V37 Stage-1 environment using 256
deterministic first-episode lanes, seed `20260914`, with observation missing
and coherent missing both zero. The selected V36 Stage-25 pass achieved hits,
full rate, mean length and view `14.7227/0.94531/1147.23/0.96478`. The V36
Stage-26 best originally used by V37 achieved only
`14.3125/0.91797/1127.57/0.96591`; V37 best, V39 best, V49 update 850 and V51
best also ranked below the Stage-25 pass on the primary full/length platform.
The structured summaries, per-lane outcomes, commands and hashes are under
`outputs/rl_sim/v54_source_selection_20260825/`.

V54 has 29 stages:

```text
1       continue the exact old V36 Stage-25 RMP-internal 12.5% polish domain
2--29   the complete V37 Stages 1--28, value-for-value and in the same order
```

Stage 1 changes no environment, reward, control, observation, physics or DR
value from V36 Stage 25. It strengthens only evidence: at least 160 updates,
a 24-update eligible hold at `conv_len/full >= 0.95/0.92`, then stochastic
block validation. There is no update cap and no forced advancement. Stage 2
therefore enters V37 Stage 1 only after the old-domain policy has reestablished
the required survival platform. Every later stage is exactly the corresponding
V37 stage; the final V54 configuration equals the final V37/V36 configuration.

V54 deliberately discards every V38--V53 intervention. It restores the V37
PPO contract: 1024 environments x 128 steps, minibatch 16384, three epochs,
LR `1e-4`, clip `0.15`, target KL `0.008`, entropy `0.0002`, gamma `0.9995`
and GAE lambda `0.99`. No transactional main-metric guard, hard-lane focus,
outcome weighting, reward rewrite or observation migration is enabled.

The launcher uses a new output directory and local offline W&B identity; it
does not modify external run `9acnp70r`. It pins GPU0 by UUID, the source and
selection hashes, XML and RMP replay evidence, rejects GPU sharing and output
overwrite, disables XLA preallocation, checks host memory and stops above
78 C. V54 remains experimental, simulation-only and non-deployable, and
inherits every unresolved real-camera, replacement-ball and formal pointwise
RMP/PD coverage blocker.

## Corrected Nominal-to-Small-to-Wide Execution DR Course V55

Status (2026-08-25): safely stopped at update 241 for reward attribution.

Profile `goal_d455_measured_qvel_rmp_vertical_v55` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v55_resume_v54_u100.sh`.
V55 resumes only the screened V54 Stage-1 archive at update 100, true step
`2194538496`, SHA-256
`5f02f9ed65afb87adfcd4b9ba6b54ded4dac1a7cf3ae9498165ac51314e4acde`.
It restores the 57-D actor, 368-D complete-DR critic, Adam moments and true
optimizer time, but starts fresh convergence history at
`rmp55_nonexecution_recovery_nominal_execution`. Curriculum-history resume,
optimizer/critic reset, observation migration, another source checkpoint,
transaction promotion and advantage reweighting must fail closed.

V54's initial Stage-25 domain already contained the complete 12.5% recovered-
RMP interval and therefore violated the intended V37 prerequisite ordering.
At the safely stopped V54 tail, rolling `conv_len/full` remained near
`0.90/0.85`; this was not a PPO numerical failure. Frozen update-100 screening
used 256 first-episode lanes, seed `20260916`, missing probability `0.15` and
coherent missing zero. The execution-domain comparison was:

| execution domain | hits | full | mean length |
| --- | ---: | ---: | ---: |
| V36 Stage 25, RMP 12.5% | 13.984 | 0.879 | 1100.6 |
| V36 Stage 23, RMP 6.25% | 13.984 | 0.875 | 1098.0 |
| V36 Stage 21 small RMP/PD/plant | 14.543 | 0.926 | 1134.5 |
| exact V55 Stage 1, execution DR disabled | 14.496 | 0.914 | 1122.9 |

Removing observation missing alone did not recover survival
(`full=0.863`, mean length `1089.7`). The large execution domain, rather than
the 13--15-hit cadence or missing observations, is therefore the first causal
course error to remove. The small and nominal conditions both preserve the
desired roughly fourteen contacts and materially improve longevity. Evidence,
per-lane outcomes and the diagnostic environment payload are under
`outputs/rl_sim/v55_primary_reward_repair_20260825/`.

V55 has exactly 28 stages and preserves the requested numbering:

```text
1       V37 Stage-1 task/ball/observation domain with no execution DR
2--19   V37 non-execution lessons with V36 Stage-21 small execution DR
20      exact conv_len=1.0/full=1.0 protected non-execution polish
21--22  recovered-RMP 6.25% bridge and polish
23--24  recovered-RMP 12.5% commit and polish
25--26  PD/plant 6.25% bridge and polish
27--28  complete V36 execution DR commit and polish
```

Stage 1 disables recovered-RMP and PD randomization and pins RMP Kp/Kd,
estimator process/measurement multipliers, acceleration weight, PD Kp/Kv,
damping and armature to `1.0`; velocity feedforward to `0.5`; target-filter
length to `10`; and output-delay offset to zero. Stage 2 restores exactly the
existing V36 Stage-21 small ranges. No 6.25% or 12.5% range is allowed before
the unchanged Stage-20 full polish has passed its rolling and block-validation
gates. Stages 2--28 otherwise match V37 value-for-value.

V55 changes no reward. Positive hit-event credit remains capped at 14 and the
mean-hit gate remains 13, representing the required 13--15 contacts over a
complete 1200-step episode. Alive, post-hit survival, apex/lift, cadence,
termination, safety and view shaping are V37-exact. Reward changes are deferred
unless nominal/small-DR training still fails to improve `conv_len/full`; any
later reward experiment requires a new profile and must protect hits, apex,
view, qvel/qacc and failure distribution.

The PPO contract remains 1024 environments x 128 steps, minibatch 16384,
three epochs, LR `1e-4`, clip `0.15`, target KL `0.008`, entropy `0.0002`,
gamma `0.9995` and GAE lambda `0.99`. The formal launcher creates a new output
directory and a new W&B online identity with `resume=never`; it never appends
to `9acnp70r` or the V54 offline run. GPU UUID, source/evidence hashes, no-
sharing, host-memory, XLA no-preallocation and 78 C gates remain mandatory.
V55 remains experimental, simulation-only and non-deployable and inherits all
unresolved real-camera, replacement-ball and formal RMP/PD coverage blockers.

The online run was stopped safely at Stage 1 update 241, true step
`2226126848`. The preserved `mjx_curriculum_last.pkl` has SHA-256
`ed02beca220da604e9e13da6fbd0905d3de8f8305d0ae4003c200fd76fe13ca7`.
Comparing the first and final 40 updates, absolute apex fell
`1.28286 -> 1.27538 m`, contact-to-apex lift fell
`0.15866 -> 0.15472 m`, hits rose `12.756 -> 14.536`, recent length fraction
fell `0.95238 -> 0.94455`, and full rate oscillated near the gate
(`0.91793 -> 0.92442`). Mean return nevertheless rose by about 10.43. This is
the frozen source for the reward analysis below; V55 must not be resumed.

## Joint Absolute-Apex/Flight-Lift Repair V56

Status (2026-08-25): bounded GPU0 reward trial completed; rejected for
insufficient height movement.

Profile `goal_d455_measured_qvel_rmp_vertical_v56` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v56_resume_v55_u241.sh`.
It resumes exactly the stopped V55 update-241 checkpoint and preserves its
57-D actor, 368-D complete-DR critic, Adam moments and true optimizer step.
Another profile/stage/step/hash, curriculum-history resume, optimizer or
critic reset, observation migration, transactional promotion and advantage
reweighting fail closed. V56 starts a fresh convergence window at
`rmp56_joint_apex_lift_repair_nominal_execution`.

The V55 Stage-1 attribution found 124 logged reward channels: total and dense
aggregates plus 122 component terms. Of those terms, 87 were identically zero
and 35 were active in this domain. Sparse
positive hit credit was correctly capped at 14, but the inherited height
contract had no effective lower correction: `low_hit_penalty_weight=0`, while
the 1.32 m `hit_height_penalty` admitted a broad 1.25--1.39 m deadband. The
absolute-apex and lift targets each had weight 0.75. Full 15+ episodes could
therefore gain dense alive/survival, compact-path and view/motion-quality
credit while accepting only a small absolute-apex loss. In the frozen V54
source population, full 15+ episodes exceeded full 14-hit return by about
2.29 despite lower apex; the largest favorable terms were dense reward
`+2.91`, including post-first-hit alive `+1.66` and post-hit survival `+0.49`.
The absolute-apex target opposed the behavior by only `-0.70`. Across full
episodes, absolute apex and hit count correlated `-0.44`.

V56 does not optimize absolute height alone. It jointly defines:

| physical quantity | V55 | V56 |
| --- | ---: | ---: |
| absolute world-z apex target | 1.32 m | 1.32 m |
| absolute target weight / sigma | 0.75 / 0.050 m | 2.5 / 0.040 m |
| one-sided absolute lower edge / weight | inactive | 1.28 m / 2500 m^-2 |
| contact-to-apex lift target | 0.175 m | 0.195 m |
| lift target weight / sigma | 0.75 / 0.040 m | 2.5 / 0.025 m |
| lift barrier interval / weight | 0.130--0.220 m / 100 | 0.175--0.215 m / 1000 |
| counted cadence target / weight | inactive | 0.390 s / 0.5 |
| ball/racket contact-z soft limits | 1.18/10.0 m | 1.18/1.16 m |
| ball/racket contact-z weights | 400/0 | 400/400 |

The two target heights imply a contact ball height of
`1.32 - 0.195 = 1.125 m`, close to the stopped source's measured 1.121 m.
Thus the policy cannot satisfy absolute apex merely by raising the whole
juggling location. A 0.195 m ballistic lift corresponds to about 0.40 s per
cycle, aligning the physical flight with 13--15 contacts over 1200 steps.
The positive hit-event cap remains 14, and no maximum-hit termination or hard
maximum-hit graduation gate is introduced.

All 28 V55 stages and their corrected execution-DR order remain present.
Control, observation, reset, physics, ball/contact DR, RMP/PD values, PPO and
all non-height reward fields are preserved. Only the three height gates are
tightened: minimum absolute apex 1.29 m and lift interval 0.175--0.215 m;
ball/racket contact-height gates remain 1.18/1.16 m. The preregistered bounded
trial uses 1024 environments for 32 updates, one stage, seed `20260919`, W&B
offline and a unique output directory. It must improve both absolute apex and
lift without raising contact height, while protecting full/length, 13--15
hits, cadence, view, racket motion, qvel/qacc, KL and failure distribution.
Return alone cannot select it.

The 32-update trial did not meet the physical-height acceptance threshold.
On a paired seed-`20260920`, 256-lane frozen screen, V56 update 32 versus the
V55 source changed absolute apex by `+0.00190 +/- 0.00084 m`, lift by
`+0.00299 +/- 0.00058 m`, and contact ball height by
`-0.00109 +/- 0.00057 m`. Full rate improved `0.91797 -> 0.92969` and mean
length changed `1128.3 -> 1133.4`, so the direction was safe and did not use
an upward contact-location shortcut. It nevertheless finished at only about
`1.2771 m` absolute apex and `0.1561 m` lift, far below the preregistered
`1.29/0.175 m` thresholds. Do not launch V56 formally or resume its trial
actor. Preserve it as evidence that the coherent height reward is correctly
directed but insufficient under the inherited low-exploration PPO contract.

V56 is experimental, simulation-only and non-deployable and inherits every
unresolved real-camera, replacement-ball and formal RMP/PD coverage blocker.

## Height-Escape PPO Trial V57

Status (2026-08-25): bounded GPU0 optimization trial completed; insufficient
height for formal promotion.

Profile `goal_d455_measured_qvel_rmp_vertical_v57` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v57_resume_v55_u241.sh`.
It restarts from the same stopped V55 update-241 checkpoint and must not
inherit the rejected V56 actor. V57 copies every one of V56's 28 environment,
reward, gate, curriculum and DR values exactly. The only changed contract is
the PPO height-escape preset: learning rate `1.5e-4`, clip `0.20`, target KL
`0.012`, entropy coefficient `0.001`, and minimum log standard deviation
`-4.0`; rollout length remains 128, epochs remain three, gamma remains
`0.9995`, and GAE lambda remains `0.99`.

This experiment tests an observed optimization blocker, not a new reward
hypothesis. V55/V56 policy log standard deviations had three dimensions at or
within numerical tolerance of the `-4.8` floor, and V56's physical improvement
stalled after update 20 even though all four new height reward components
improved in paired attribution. The 64-update offline trial must move absolute
apex and lift together to at least `1.29/0.175 m`, retain contact ball/racket
heights below `1.18/1.16 m`, keep hits in the intended 13--15 range, and avoid
material full/length, view, motion, qvel/qacc, KL or failure-distribution
regression. A higher stochastic return or policy entropy is not acceptance.

The 64-update trial remained below the `1.29/0.175 m` acceptance threshold,
so V57 may not be launched formally. It nevertheless produced a materially
better platform. On paired seed-`20260922`, 128-lane frozen evaluation, the
last policy achieved hits/full/length `14.969/0.97656/1177.4`; absolute
apex/lift/contact ball z were `1.28375/0.16212/1.12163 m`. Versus V55, apex
and lift changed `+8.30/+7.60 mm` while contact changed only `+0.69 mm`.
Thus exploration improved survival and both intended height coordinates
without primarily shifting contact height, but event-only lift learning was
too slow. V58 selects the complete update-60 archive rather than the
metadata-poor update-64 last file. V57 remains experimental, simulation-only
and non-deployable and inherits every unresolved real-camera,
replacement-ball and formal RMP/PD coverage blocker.

## Short-Horizon Joint Apex/Lift Credit V58

Status (2026-08-25): rejected bounded GPU0 reward-timing trial.

Profile `goal_d455_measured_qvel_rmp_vertical_v58` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v58_resume_v57_u60.sh`.
It resumes the V57 update-60 archive at true step `2233991168`, SHA-256
`0880235d472e8c599f2711707f148409a99251600f8bd15fc9a3ed97d932c8a2`,
with actor, 368-D critic, Adam moments and optimizer time intact. It starts a
fresh V58 convergence window. Another source/profile/stage/step/hash,
curriculum-history resume, optimizer/critic reset or observation migration
fails closed.

V58 changes one reward component only. During the first `0.080 s` after each
of the first 14 counted contacts, while the ball is ascending, it adds a
weight-40 dense term:

`joint_score = absolute_apex_target_score * contact_to_apex_lift_target_score`

The absolute score retains target/sigma `1.32/0.040 m`; the lift score retains
target/sigma `0.195/0.025 m`. A product is used deliberately: raising apex and
contact together retains lift but loses absolute score, while attaining lift
at a low world height loses absolute score. Hits 15+ receive no joint credit.
The existing contact-edge ball/racket height penalties and limits remain
active. Every other V57/V56 reward, gate, environment, control, observation,
physics, DR, curriculum-order and PPO value is unchanged.

The 48-update trial was rejected. In paired 128-lane seed-`20260924` frozen
evaluation, V57-u60 achieved hits/full/length `15.289/0.97656/1177.7` and
absolute apex/lift/contact ball z `1.28233/0.15818/1.12415 m`; V58-u48
achieved `14.508/0.92188/1131.0` and `1.29462/0.16559/1.12903 m`. Thus the
dense credit raised absolute apex, but physical lift still missed 0.175 m,
contact rose 4.88 mm, and full rate regressed 5.47 percentage points. Do not
launch V58 formally or resume its actor. V58 remains experimental,
simulation-only and non-deployable.

## Vertical-Strike-Antagonist Ablation V59

Status (2026-08-25): rejected bounded GPU0 single-component ablation.

Profile `goal_d455_measured_qvel_rmp_vertical_v59` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v59_resume_v57_u60.sh`.
It restarts from the same complete V57 update-60 archive at true step
`2233991168`, SHA-256
`0880235d472e8c599f2711707f148409a99251600f8bd15fc9a3ed97d932c8a2`,
with actor, critic and Adam moments intact and a fresh V59 convergence window.
It must not inherit a V58 checkpoint or resume curriculum history.

V59 retains V58's 80 ms weight-40 joint absolute-apex/lift dense term and
changes exactly one inherited reward field:
`racket_up_drift_penalty_weight: 6.0 -> 0.0`. That term is the only active
dense component that explicitly penalizes upward racket velocity whenever
the racket is above its anchor. All event rewards, alive/survival credit,
contact-height anti-cheat barriers, gates, PPO settings, control,
observations, reset/physics, all 28 curriculum stages, and the Stage-1
nominal/Stage-20 polish/Stage-21 execution-DR widening order remain exact.

The 48-update trial was rejected. Paired seed-`20260926`, 128-lane frozen
evaluation measured V57-u60 at abs/lift/contact
`1.28319/0.15904/1.12415 m`, full `0.92188`, length `1125.7`; V59-u48 reached
`1.29202/0.16581/1.12622 m`, full `0.92969`, length `1136.5`. The ablation
improved height and longevity safely but lift remained 9.2 mm below its lower
edge. The up-drift penalty is a real local antagonist but not the root cause.
Do not launch V59 formally or resume its actor.

## Completion-Conditioned Physical Objective V60

Status (2026-08-25): rejected bounded GPU0 objective-alignment trial.

Profile `goal_d455_measured_qvel_rmp_vertical_v60` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v60_resume_v57_u60.sh`.
It restarts from the same complete V57 update-60 archive at true step
`2233991168`, SHA-256
`0880235d472e8c599f2711707f148409a99251600f8bd15fc9a3ed97d932c8a2`.
It preserves actor, 368-D critic and Adam state, starts a fresh convergence
window, and rejects V58/V59 sources, curriculum-history resume, state resets,
observation migration or a mismatched profile/stage/step/hash.

V60 copies V57, not the rejected V58/V59 actors or rewards. It adds one
weight-120 terminal component at the true 1200-step truncation. Credit is
zero unless total counted hits are 13--15. For eligible full episodes:

`completion_score = mean_abs_apex_score(first 14) * mean_lift_score(first 14)`

More precisely, the two Gaussian scores are evaluated on the first 14
rewardable contacts' episode means, with target/sigma `1.32/0.040 m` for
absolute world-z apex and `0.195/0.025 m` for contact-to-apex lift, then
multiplied. Hits 15+ never enter those means; 16+ total hits receive zero
terminal credit. Raising contact and apex together cannot improve lift score,
and obtaining either physical coordinate without the other cannot maximize
the product.

Every V57 dense/event term—including the weight-6 up-drift penalty—alive and
survival rewards, contact-height barriers, PPO, gates, control, observations,
physics, all 28 stages, Stage-1 nominal execution, Stage-20 polish and
Stage-21 execution-DR widening are unchanged. The default 48-update trial
uses seed `20260927`, W&B offline and a unique directory. Frozen acceptance
requires abs/lift `>=1.29/0.175 m`, safe contact height, 13--15 hits and no
material longevity, view, motion, qvel/qacc, KL or failure regression. Only a
passing screen would have permitted no-cap V60 launch in `pp_gpu0`. The trial
was rejected. Paired seed-`20260928` frozen evaluation held full at `0.92188`
and improved length by 9.5 steps, but V60-u48 versus V57-u60 changed
abs/lift/contact by `+10.40/+4.34/+6.06 mm`, ending at
`1.29399/0.16639/1.12760 m`. The terminal term protected longevity but did
not supply enough local lift credit. Do not launch V60 formally or resume its
actor.

## Direct Reflected-Velocity Continuation V61

Status (2026-08-25): preregistered bounded GPU0 local-credit trial.

Profile `goal_d455_measured_qvel_rmp_vertical_v61` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v61_resume_v57_u60.sh`.
It restarts from the exact V57 update-60 checkpoint/hash registered for V58
and does not inherit V58, V59 or V60 actors. It retains V60's terminal
completion-conditioned physical objective, then enables the existing
paper-derived pseudo-Huber adaptive reflected-velocity penalty with weight 5,
vertical sigma `0.20 m/s`, and XY sigma `100 m/s`. The large XY sigma makes
the added gradient effectively vertical-only.

For each of the first 14 rewardable contacts, the desired outgoing vertical
velocity is computed from the hit position, gravity and the 1.32 m absolute
apex target. Unlike apex-only outcome credit, it acts at the contact that
caused the flight. V57's absolute-apex and physical-lift targets plus V60's
full/13--15/product term remain active, so satisfying only outgoing velocity,
raising contact height, or failing early cannot maximize the combined reward.
All other rewards, gates, PPO, control, observations, physics, 28 stages and
nominal-small-then-wide DR order remain unchanged.

V61 was stopped at update 17 before a frozen selection screen. Its late
training points stayed near absolute apex/lift `1.284/0.160--0.161 m` with
full near `0.90--0.92`; they did not establish the required direction.
Formula inspection then found that the existing adaptive target computes
desired vertical velocity from delayed confirmation `bpos`, while V25+
physical lift uses cached contact-edge height. Enabling that term before its
coordinate contract was resolved was not justified. V61 is rejected and
incomplete: do not launch it formally, resume its actor or treat it as a
completed causal ablation.

## Existing-Reward-Only Full-14 Rebalance V62

Status (2026-08-25): preregistered bounded GPU0 trial; no formal run yet.

Profile `goal_d455_measured_qvel_rmp_vertical_v62` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v62_resume_v57_u60.sh`.
It resumes the selected V57 update-60 archive at true step `2233991168`,
SHA-256
`0880235d472e8c599f2711707f148409a99251600f8bd15fc9a3ed97d932c8a2`,
with actor, 368-D critic, Adam moments and optimizer time intact. Another
profile/stage/step/hash, curriculum-history resume, optimizer/critic reset or
observation migration fails closed. V58--V61 checkpoints are prohibited.

The complete V55 formula audit covers all 122 logged component terms, not
only nonzero correlations. It found 35 active and 87 inactive terms and is
recorded under
`outputs/rl_sim/v55_reward_rootcause_20260825/EXISTING_REWARD_FUNCTION_CONTRACT_ANALYSIS.md`
with machine-readable formula/mask/config inventory
`ALL_REWARD_COMPONENT_FORMULA_INVENTORY.csv`. The earlier V58/V60 additions
preceded delivery of that formula audit and are retained only as default-zero
negative evidence.

The rise/regression analysis gives a sharper failure signature. V55's best
absolute-apex window at updates 72--81 measured length/full/hits/absolute
apex/lift `0.94791/0.92572/14.567/1.28536/0.15915`. Its final 40 updates
measured `0.94455/0.92442/14.536/1.27538/0.15472`. Cadence changed only
`0.35874 -> 0.35960 s` and return fell, so the 10.0/4.4 mm height regression
did not buy more hits or a higher objective. The components that improved
while height fell were led by dense post-hit survival and the direct up-drift
penalty; V59 independently confirmed the latter as antagonistic.

The selected V57 checkpoint already contains the requested behavior as a
minority mode. On 128 frozen lanes at seed `20260922`, full exactly-14
episodes (n=19) measured cadence/absolute apex/lift/contact ball z/return
`0.38326 s/1.29278 m/0.17544 m/1.11734 m/131.35`. Full 15+ episodes (n=98)
measured `0.35643 s/1.27893 m/0.15769 m/1.12124 m/104.06`. V56/V57's
existing factorized rewards therefore rank the desired physical mode about
27.28 return higher; a new reward function is not needed before testing
relative scale and PPO credit.

V62 copies V57's complete 28-stage environment and changes only seven
existing reward fields:

| field | V57 | V62 |
| --- | ---: | ---: |
| `racket_up_drift_penalty_weight` | 6.0 | 0.0 |
| `center_flat_hit_reward_weight` | 2.4 | 1.2 |
| `hit_count_floor_reward_weight` | 1.5 | 0.5 |
| `hit_reward_base` | 1.0 | 0.75 |
| `hit_cadence_target_interval` | 0.390 s | 0.390 s |
| `hit_cadence_sigma` | 0.045 s | 0.035 s |
| `hit_cadence_reward_weight` | 0.5 | 2.0 |

V56's absolute-apex target, physical-lift target, contact-edge ball/racket-z
barriers and gates remain unchanged. Dense alive/survival, miss/limit
terminal costs, view and motion-quality protection, PPO environment, control,
observations, physics, reset distributions and every DR endpoint/order remain
V57-exact. V58's `post_hit_joint_apex_lift_reward`, V60's
`full_episode_joint_height`, and V61's adaptive reflected-velocity weight are
all exactly zero. Stage 1 disables execution RMP/PD/plant DR, Stage 20 remains
`complete_nonexecution_full_episode_polish`, and only Stage 21 begins wide
execution DR.

The initial candidate uses the V57 height-escape PPO settings, 1024
environments, rollout 128, three epochs, seed `20260930`, one stage, at most
64 updates, W&B offline and a unique output directory. Frozen acceptance
requires mean hits in 13--15, full/length protection, absolute apex/lift at
least `1.29/0.175 m`, contact-z protection, counted interval near the desired
0.38--0.40 s mode, stable view/motion quality, zero qvel exceedance and qacc
exceedance at most 0.01. Multiple periodic checkpoints must be screened; last
or higher-return alone cannot select the source.

Only a passing checkpoint permits the same no-cap 28-stage V62 course in
tmux `pp_gpu0` with a unique output directory, W&B online, a brand-new run ID
and `resume=never`. V62 remains experimental, simulation-only and inherits
all unresolved real-camera, replacement-ball and formal RMP/PD blockers.

## Robust V31 Source, Exact-Reentry Diagnostic V63

Status (2026-08-25): completed negative bounded GPU0 diagnostic; do not resume.

Profile `goal_d455_measured_qvel_rmp_vertical_v63` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v63_resume_v31_stage23.sh`.
It resumes the V31 Stage-23 polish archive at true step `2111569920`, SHA-256
`93a4002fe579dd34b1768b058c7e81f9c9def937afbfea64cac182a5b98498dc`.
The source has a 57-D actor, 221-D critic and complete Adam state. V63 starts
fresh convergence history and rejects another profile/step, curriculum-state
resume, optimizer/critic reset and observation migration.

This source was selected from paired frozen 128-lane screens at seeds
`20260932` and `20260933`, not from return or a single aggregate. The complete
candidate summaries and per-episode evidence are under
`outputs/rl_sim/v63_robust_source_selection_20260825/`. The principal combined
comparison is:

| source | full worst/mean | length worst/mean | mean hits | full 13--15 | full 16+ | abs apex | lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V17 Stage 13 | 0.906/0.934 | 1108.0/1136.2 | 15.152 | 0.195 | 0.699 | 1.2708 | 0.1380 |
| V28 Stage 17 | 0.883/0.914 | 1093.2/1125.2 | 14.242 | 0.512 | 0.395 | 1.2928 | 0.1635 |
| V29 Stage 21 | 0.906/0.938 | 1120.9/1147.8 | 14.344 | 0.609 | 0.316 | 1.2939 | 0.1676 |
| **V31 Stage 23** | **0.922/0.941** | **1124.5/1144.6** | **14.324** | **0.621** | **0.309** | **1.2966** | **0.1694** |
| V37 best | 0.930/0.941 | 1131.7/1143.8 | 14.715 | 0.375 | 0.531 | 1.2781 | 0.1548 |
| V57 update 60 | 0.891/0.910 | 1101.7/1123.1 | 14.266 | 0.441 | 0.449 | 1.2836 | 0.1600 |
| V62 update 20 | 0.898/0.918 | 1112.4/1131.3 | 14.332 | 0.480 | 0.422 | 1.2888 | 0.1614 |

The V31 source is the most balanced robust base, but it is not the final
policy: its worst-seed full fraction is still `0.922` and its mean lift is
only `0.1694 m`. V16 itself is not resumed because its source-to-current
configuration has 79 normalized environment-field changes and critic growth
from 221 to 368; it is evidence that a large direct course jump is unsafe,
not a better direct continuation.

V63 contains one stage, `rmp63_v31_stage23_exact_reentry`. The implementation
constructs V31 and copies its Stage 23 value-for-value; only the stage name and
notes change. Therefore reward, graduation gates, 6.25% internal-RMP DR,
ball/contact DR, reset distribution, control, actor/critic observations and
physics have zero configuration difference from the selected checkpoint.
The historical checkpoint predates 12 later-added configuration fields; each
is absent in the pickle and hydrates to its current backward-compatible class
default (eight zero/inactive values, two false critic flags and the two true
legacy per-joint RMP flags). There is no non-default semantic difference.
The original PPO contract is also frozen: 1024 environments, rollout 128,
minibatch 16384, three epochs, learning rate `1e-4`, gamma `0.9995`, GAE
`0.99`, clip `0.15`, target KL `0.008`, entropy `0.0002`, minimum log standard
deviation `-4.8`, value coefficient `0.5` and gradient norm `0.5`.

The preregistered first run uses seed `20260934`, one stage, at most 32
updates, per-update archives, W&B offline with `resume=never`, GPU0 UUID
`GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`, no preallocation and the 78 C
guard. Frozen selection must compare multiple checkpoints at seeds `20260932`
and `20260933`. Acceptance requires mean/worst full at least `0.93/0.90`,
mean/worst length at least `1135/1110`, mean hits in 13--15, at least `0.55`
of full episodes in 13--15, absolute apex at least `1.29 m`, lift at least
`0.165 m`, protected contact height and view, zero qvel exceedance and qacc
exceedance at most `0.01`.

The 32-update run is preserved at
`outputs/rl_sim/measured_qvel_rmp_vertical_v63_gpu0_seed20260934_20260825_v31_stage23_zero_gap_u32/`.
Its final 24-update online window measured hits/length fraction/absolute
apex/lift `13.4385/0.90136/1.29502/0.16728`, but it did not meet the complete
convergence hold. Paired frozen source/update-10/update-20/update-30/update-32
screens in the exact V31 Stage-23 environment selected update 20, whose
two-seed combined full/length/hits were `0.875/1109.63/13.883`, full 13--15
fraction `0.5938`, apex/lift `1.29498/0.16738 m` and view `0.9456`. It fails
the preregistered survival and view thresholds. Update 30 and last regressed
further. V63 is rejected; never select its last checkpoint.

The first source-selection table above used V62's nominal-execution
environment. That was valid for comparing policies in V62 Stage 1 but not for
selecting a source for V31 Stage-23 exact reentry, which applies 6.25%
internal-RMP DR. The corrected paired screens below use the actual target
environment for every 221-D candidate:

| source | full worst/mean | length worst/mean | mean hits | full 13--15 | full 16+ | view | abs/lift (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V16 Stage 13 | 0.859/0.871 | 1074.6/1085.4 | 14.074 | 0.184 | 0.722 | 0.956 | 1.2670/0.1372 |
| V17 Stage 13 | 0.883/0.883 | 1093.8/1098.0 | 14.473 | 0.190 | 0.748 | 0.956 | 1.2695/0.1374 |
| V24 best | 0.805/0.805 | 1048.4/1051.5 | 13.695 | 0.291 | 0.699 | 0.935 | 1.2954/0.1493 |
| V28 Stage 17 | 0.852/0.859 | 1075.9/1079.0 | 13.520 | 0.532 | 0.445 | 0.943 | 1.2905/0.1613 |
| **V29 Stage 21** | **0.875/0.891** | **1103.6/1121.1** | **14.039** | **0.583** | **0.404** | **0.965** | **1.2932/0.1668** |
| V31 Stage 23 | 0.844/0.863 | 1078.9/1094.2 | 13.660 | 0.584 | 0.403 | 0.946 | 1.2935/0.1665 |
| V63 update 20 | 0.867/0.875 | 1098.4/1109.6 | 13.883 | 0.594 | 0.393 | 0.946 | 1.2950/0.1674 |

## Corrected V29 Exact-Reentry Diagnostic V64

Status (2026-08-25): completed negative bounded GPU0 diagnostic; do not resume.

Profile `goal_d455_measured_qvel_rmp_vertical_v64` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v64_resume_v29_stage21.sh`.
It resumes V29 Stage 21 at true step `2087190528`, SHA-256
`8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`,
with its 57-D actor, 221-D critic, optimizer time and all Adam moments.

V64 has one stage, `rmp64_v29_stage21_exact_reentry`, copied exactly from V29
Stage 21 except for its identity and notes. Thus its source-to-Stage-1
configuration distance is zero. V29 Stage 21 to V31 Stage 23 differs in only
seven internal-RMP fields: Kp/Kd, estimator process/measurement, velocity
feedforward and acceleration-weight multiplier ranges, plus output-delay
offset. V64 deliberately does not apply those seven widenings; a future bridge
must be separately versioned after source retention is established.

The PPO contract is source-exact: 1024 environments, rollout 128, minibatch
16384, three epochs, learning rate `1e-4`, gamma `0.9995`, GAE `0.99`, clip
`0.15`, target KL `0.008`, entropy `0.0002`, minimum log standard deviation
`-4.8`, value coefficient `0.5` and gradient norm `0.5`. The first run uses
seed `20260935`, one stage, at most 32 updates, W&B offline and a new identity.
The same two-seed frozen acceptance contract as V63 applies. Reward changes,
critic growth, transaction guards, outcome/hard-lane weighting and PPO scans
are prohibited. If no source or periodic checkpoint passes, the next work is
a constrained-MDP/risk-objective analysis; PPO hyperparameters remain last.

The run is preserved at
`outputs/rl_sim/measured_qvel_rmp_vertical_v64_gpu0_seed20260935_20260825_V29_Stage21_zero_gap_u32/`.
The source remained best. Its mean/worst full and length were
`0.89453/0.875` and `1117.48/1109.83`; V64 update 30 measured only
`0.87891/0.875` and `1112.25/1105.13`, while last measured `0.875/0.86719`
and `1107.57/1103.46`. No periodic checkpoint passed. V64 is rejected and
must not be resumed.

Full and failed V29-source episodes received mean return `110.66/-1.24`, so
the scalar reward already ranks full behavior correctly. Source failures were
long-tail early episodes: 27/256, mean length `417.6`, dominated by 11
racket-too-high and 10 ball-too-low terminations. V64 PPO was numerically
healthy (final-24 exact KL mean/max `0.00492/0.00789`, EV `0.896`, qvel
exceedance zero and qacc exceedance `0.00470`) while survival regressed.

This closes source selection, zero-gap course reentry and PPO numerical-fault
checks. The next design is an opt-in constrained MDP with separate
failure/remaining-length cost critics and adaptive dual variables; it is not
another fixed reward weight or PPO scan. It must first use the exact V29
Stage-21 domain and keep the seven-field RMP bridge out of scope. The complete
derivation, evidence tables and preregistration are in
`outputs/rl_sim/v63_robust_source_selection_20260825/V63_V64_SOURCE_SELECTION_AND_CONSTRAINED_OBJECTIVE_DECISION.md`.

## V29 Stage-21 Constrained-MDP Continuation V65

Status (2026-08-25): completed negative bounded GPU0 experiment; do not resume
V65 and do not start formal training.

Profile `goal_d455_measured_qvel_rmp_vertical_v65` is launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v65_cmdp_resume_v29_stage21.sh`.
It resumes the pinned V29 Stage-21 checkpoint at true step `2087190528`,
SHA-256 `8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`.
Stage `rmp65_v29_stage21_cmdp_survival` differs from V29 Stage 21 only in name
and notes. The 57-D actor, 221-D reward critic, environment reward, gates,
control, physics, reset distribution, ball/contact DR and all PPO values are
unchanged.

V65 changes the optimization state explicitly. It appends a zero-output
six-channel cost critic and zero Adam leaves while preserving every source
actor/reward-critic parameter, Adam moment and Adam time. Its undiscounted
episodic costs and upper bounds are:

| cost | episode value | bound |
| --- | --- | ---: |
| failure | `1[true termination]` | `0.07` |
| shortfall | `1[true termination] * (1200-T)/1200` | `65/1200` |
| ball low/high | corresponding terminal indicator | `0.02` each |
| racket low/high | corresponding terminal indicator | `0.02` each |

Each channel has an independent cost value estimate. Cost GAE uses gamma
`1.0`, lambda `0.99`; the cost-value coefficient is `0.5`. The actor advantage
is the normalized reward advantage minus projected nonnegative duals times
independently normalized cost advantages, followed by a final normalization.
Duals initialize at `[0.25,0.25,0.10,0.10,0.10,0.10]`, update at rate `0.1`
from completed-episode empirical costs and are clipped to `[0,2]`. They and
all cost-head optimizer state are checkpointed and logged. Direction-specific
channels prevent satisfying aggregate survival by merely exchanging high and
low terminal modes.

The CMDP path is default-off and restricted to V65. V65 prohibits reward,
PPO, observation, transaction-guard and legacy advantage-focus changes. It
also keeps the V31 seven-field internal-RMP widening out of scope. A CUDA
GPU0 smoke with 16 environments and one update completed successfully at
`outputs/rl_sim/v65_cmdp_20260825/smoke_gpu0_n16_u1_escalated/`; it verified
57/221/7 dimensions, cost-head/dual checkpoint persistence, preserved Adam
time and scalar CMDP telemetry. The smoke is compilation evidence only, not
policy evidence.

The preregistered bounded run uses 1024 environments, rollout 128, minibatch
16384, three epochs, seed `20260936`, at most 32 updates and W&B offline with
a new identity. Frozen deterministic first-episode screens use seeds
`20260932` and `20260933`. A checkpoint may proceed to no-cap `pp_gpu0`
training only if mean/worst full reach `0.93/0.90`, mean/worst length reach
`1135/1110`, mean hits remain 13--15, at least `0.55` of full episodes have
13--15 hits, absolute apex is at least `1.29 m`, lift at least `0.165 m`, view
at least `0.96`, qvel exceedance zero and qacc exceedance at most `0.01`.
Failure preserves the result and blocks formal launch.

The exact V29 Stage-21 screens retained the source's 15% camera/view missing
probabilities. Update 10 was the least-regressed V65 candidate, with
mean/worst full `0.8945/0.8906`, mean/worst length `1115.0/1113.6`, mean hits
`13.953`, full 13--15 fraction `0.6070`, view `0.9640` and absolute apex/lift
`1.2956/0.1683 m`. It fails mean/worst full and mean length. The source was
`0.8945/0.8828` full and `1111.2/1110.5` length; therefore update 10 improved
only the worst seed and length slightly, not aggregate survival. Update 30
regressed to `0.8750/0.8672` full, and no later checkpoint passed.

The 256 paired source episodes ended ball-low/racket-high `8/12` times;
update 10 changed these to `11/9`, so V65 exchanged height-failure direction
rather than satisfying the aggregate failure constraint. Training remained
numerically healthy (reward-critic EV `0.899`, exact KL mean `0.00658`, qvel
exceedance zero, qacc exceedance about `0.00508`), but cost-critic EV after
warmup stayed near zero: failure `-0.0048`, shortfall `-0.0020`, ball-low
`0.0067`, ball-high `-0.0365`, racket-low `-0.0054`, racket-high `0.0428`.
Failure/shortfall duals rose to `0.448/0.370` without producing calibrated
long-horizon risk advantages.

The complete report and hashed exact/zero-missing evidence are at
`outputs/rl_sim/v65_cmdp_20260825/V65_CMDP_BOUNDED_VALIDATION_REPORT.md`.
The zero-missing screens are an ablation, not zero configuration distance;
never substitute them for the 15% missing V29 contract. A future monotonic
successor should pretrain cost values from complete frozen-episode labels and
pass held-out risk calibration before actor training. Increasing terminal
reward/dual weight or merely running V65 longer is not authorized by this
result.

## V29 Stage-21 Bimodal-Failure Causal Diagnostic

Status (2026-08-25): completed frozen GPU0 diagnosis. This is evidence, not a
V66 curriculum profile. CMDP remains disabled; do not resume V65 or start a
formal `pp_gpu0` training run from this diagnostic.

The frozen actor and environment are the V29 Stage-21 checkpoint at true step
`2087190528`, SHA-256
`8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`.
Seeds `20260932/20260933` supplied 128 first episodes each. The full-DR
baseline was strongly bimodal: 227/256 episodes reached 1200 steps, while 29
failures averaged 395.9 steps and none occupied steps 1000--1199. Terminations
were ball-low/racket-high/racket-low/ball-X/ball-Y `13/11/3/1/1`.

Exact same-graph re-execution proved that failure identities are not fixed.
Across the baseline plus three repeats, 205 lanes were always full, 14 always
failed and 37 changed outcome. Aggregate exact-replay full was `0.89063`.
Changing only temporal RNG averaged `0.88281`; four stochastic-action repeats
averaged `0.88574`. Gaussian exploration has no consistent net effect and is
not the primary platform cause. Long MJX contact rollouts amplify tiny
numerical and sensing perturbations near a closed-loop separatrix, so one
deterministic lane result must not be treated as an immutable label.

Paired counterfactuals establish a DR-dependent robustness-margin problem:

| frozen counterfactual | full | length | fail->full / full->fail |
| --- | ---: | ---: | ---: |
| V29 full DR | 0.88672 | 1108.91 | baseline |
| nominal ball/contact DR | 0.94531 | 1166.77 | 25 / 10 |
| nominal recovered-RMP/PD DR | 0.92969 | 1144.42 | 19 / 8 |
| nominal static observation/racket-mount DR | 0.93750 | 1144.30 | 20 / 7 |
| all episode-constant DR nominal | **1.00000** | **1200.00** | **29 / 0** |
| temporal sensing/missing/target-hold off | 0.94922 | 1154.45 | 21 / 5 |
| centered reset with full DR | 0.90625 | 1136.20 | 15 / 10 |

The joint nominal condition retains the target/reset support and per-step
randomness; mount nominalization necessarily changes the corresponding
physical contact geometry. Centering the reset did not outperform paired
branch-exchange noise. Therefore do not narrow reset/target support, call the
registered DR invalid, or claim an impossible physical population. Three
baseline lanes were persistently DR-sensitive in the limited repeats, but
their count of outer-five-percent DR features was no greater than expected
from the 117 inspected fields. They are hard combinations inside the current
support.

The failed-cycle trace supplies a proximal learning signal that V65 lacked.
At 3446 full-episode hits versus 26 failed-episode final hits, mean intercept
racket-XY error was `0.00886 -> 0.04928 m`; `92.3%` of failed final hits but
only `1.25%` of full hits exceeded the existing `0.025 m` intercept radius.
Contact-point racket VXY exceeded `0.25 m/s` in `61.5%` versus `13.7%`, action
norm exceeded `1.4` in `61.5%` versus `1.86%`, and racket-Z relative to anchor
exceeded `0.14 m` in `38.5%` versus `0.35%`. Median final-hit-to-termination
delay was 86 steps for ball-low and 69 for racket-high. Ball-low failures then
missed/underpowered the next contact; racket-high/low failures drifted
monotonically out of the vertical recovery region.

Do not repeat V45--V53 success anchoring, single-rollout hard indicators,
scalar completion rewards, fixed transactions, or V65 CMDP cost critics. A
future monotonically versioned bounded experiment must instead replicate each
episode-constant DR/reset tuple across independent temporal/action draws,
estimate tuple failure probability, stratify the ball/contact,
observation/mount and RMP/PD families before temporal randomness, and give
empirically hard tuple groups fixed episode-level sampling mass. A code audit
found that `pre_hit_intercept_*` is active only while `hit_count <= 0`, while
27/29 audited failures occur after the first hit. The isolated reward
candidate must therefore use the existing recurrent
`descending_intercept_excess_penalty` with a 0.025 m zero-loss radius; a
failed-contact event candidate must be separate. DR intervals, reset
support, PPO parameters and completion reward stay unchanged in either
ablation. Formal training remains blocked until a bounded repeated-pair screen
passes the complete 1200-step, full/length, 13--15 hit, apex/lift, view,
motion and safety contract.

The complete derivation, exact McNemar comparisons, hashes, lane
classifications, success/failure event table and sampled traces are under
`outputs/rl_sim/v66_bimodal_failure_analysis_20260825/`, with the primary
report `V29_BIMODAL_FAILURE_CAUSAL_REPORT.md`.

## V29 Replicated-Tail Recovery Continuation V66

Status (2026-08-26): bounded negative result; CMDP remained disabled and V66
must not be launched formally. Profile
`goal_d455_measured_qvel_rmp_vertical_v66` resumes
only V29 Stage 21 at true step `2087190528`, SHA-256
`8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`.
Its launcher is
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v66_resume_v29_stage21.sh`.

The code audit corrected the earlier pre-hit proposal. The pre-hit terms are
active only at `hit_count <= 0`, while 27/29 audited failures occur later.
Stage `rmp66_post_hit_intercept_recovery_bridge` therefore changes only the
existing recurrent descending-intercept excess penalty to weight `2.4`,
radius `0.025 m`, sigma `0.050 m`; its existing time horizon stays `0.55 s`.
V29 DR intervals, correlations, reset/observation/control contracts, PPO,
completion rewards, CMDP and outcome advantage weighting remain unchanged.
Its `conv_len/full` gates are `0.90/0.84`, minimum 64 updates and 12-update
hold. The penalty is zero for the measured successful-contact core. At the
failed median error (`~0.049 m`) it contributes about 11% of the positive
post-first-hit alive tick; at `0.10 m` it becomes comparable, producing a
local recovery gradient without making termination credit dominate PPO.

Stage `rmp66_replicated_hard_tuple_consolidation` changes no environment
field. It assigns 10% of actual eligible episode resets to six complete JAX
reset keys corresponding to the replicated tuples with <=4/18 successes.
Each actual eligible episode reset independently receives the fixed 10%
mixture probability, so short hard episodes cannot alter that per-reset
probability. The stage
uses `conv_len/full=0.92/0.88`, minimum 96 updates and a 16-update hold. Zero
success in 18 draws is explicitly empirical hardness, not proof of physical
unreachability.

The preregistered bounded run is 32 updates at the source PPO shape. Frozen
two-seed repeated screens require aggregate full `>=0.90`, mean length
`>=1125`, worst-seed full `>=0.87`, mean hits 13--15, full 13--15 fraction
`>=0.55`, absolute apex/lift `>=1.28/0.145 m`, view `>=0.96`, zero qvel
exceedance, qacc exceedance `<=0.01`, and a material six-tuple improvement.
The formal run must restart the pinned V29 source, use both stages, no update
cap, a new output/W&B identity and tmux `pp_gpu0`.

The 32-update bounded run completed normally at its cap. Update 30 was the
least-regressed frozen candidate: deterministic two-seed full/worst full
`0.90234/0.89844`, mean/worst length `1124.82/1123.05`, hits `14.117`, full
13--15 fraction `0.5671`, view `0.9678`, and absolute apex/lift
`1.2948/0.1670 m`. It missed the mean-length threshold by `0.18` step and was
therefore subjected to the registered repeated screen rather than rounded up.
Across baseline, same-graph, four process and four stochastic-action draws,
update 30 achieved only full `0.89414`, mean length `1113.55`, and hard-six
success `8/60=13.33%` versus source `12/108=11.11%`. This fails aggregate
full, length and material hard-group improvement. Preserve V66 as a negative
result; do not resume update 30 or reinterpret its healthy PPO values as
policy acceptance. Complete evidence and hashes are in
`outputs/rl_sim/v66_dr_reachability_audit_20260826/V66_BOUNDED_VALIDATION_AND_V67_DECISION.md`.

## V29 Half-Spin Replicated-Tail Continuation V67

Status (2026-08-26): bounded trial accepted and formal offline-evidence
training completed. Profile `goal_d455_measured_qvel_rmp_vertical_v67` launches only via
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v67_resume_v29_stage21.sh`.
It restarts the same pinned V29 Stage-21 source at true step `2087190528` and
SHA-256
`8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`;
V66 trial checkpoints are prohibited.

V67 Stage 1, `rmp67_half_spin_support_bridge`, restores the original V29
reward and changes only ball-spin support from
`+/-25,+/-25,+/-20 rad/s` to `+/-12.5,+/-12.5,+/-10 rad/s`. Solref and every
other DR field, sampling correlation, reset, observation, control, reward and
PPO value remain V29-exact. The stage uses `conv_len/full=0.92/0.88`, minimum
64 updates and a 12-update hold. The narrower spin interval is an empirically
better unmeasured candidate, not replacement-ball hardware coverage.

V67 Stage 2, `rmp67_half_spin_hard_tuple_consolidation`, changes no
environment field and gives each eligible episode reset a fixed 10%
probability of one of the same six registered hard keys. Its gates are
`conv_len/full=0.92/0.90`, minimum 96 updates and a 16-update hold. CMDP,
outcome advantage weighting, completion bonuses and V66's intercept reward
remain disabled. Neither stage lowers `conv_len` below `0.90`.

The causal basis is the frozen two-seed four-repeat support audit: half spin
reached full `0.92480`, mean length `1146.58` and hard-six success `16/24`,
while half solref reached `0.91602/1133.82/12-of-24`. This authorizes testing
half spin first, not changing solref simultaneously. A bounded 32-update run
must preserve stable PPO/safety and the exact half-spin profile contract
before formal launch. Formal training must restart V29, use both stages with
no update cap, create a new output/W&B identity and run in tmux `pp_gpu0`.

The formal run is
`outputs/rl_sim/measured_qvel_rmp_vertical_v67_gpu0_seed20260939_20260826_V29_Stage21_half_spin_replicated_tail_formal1`
with seed `20260939`, physical GPU0 UUID
`GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`, W&B offline ID `v67g0a1`,
1024 environments, rollout 128 and batch 131072. Stage 1 graduated after 75
updates, therefore satisfying the user-required observation beyond 64
updates. Stage 2 graduated at update 314 after satisfying its minimum-96 and
16-update strict hold. The final checkpoint is at true step `2138177536`,
Adam time `441976`, SHA-256
`1da7940585d1b22d8b7bbe4974b1e700bce645cc1a7983b51b68192e3431c2e4`.
Its final 24-update window had hits `13.8536`, length fraction `0.93270`, full
`0.90104`, absolute apex `1.29067 m` and lift `0.16719 m`. This is an accepted
experimental source, not a released or hardware-covering model.

## Complete Non-Execution-First / Wide-Execution-DR Course V68

Status (2026-08-26): exact Stage-4 online recovery active in `pp_gpu0` with
1024 environments.

Profile `goal_d455_measured_qvel_rmp_vertical_v68` initially launches only via
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v68_resume_v67_final.sh`.
It resumes only the completed V67 final checkpoint above. The validator pins
source profile, true step, dimensions `57/221/7`, Adam time, PPO settings,
mass range, half-spin support, independent-per-joint source sampling and
absence of CMDP state. Curriculum history starts fresh. The 57-D actor and
all actor optimizer moments remain unchanged; append-only critic migration
preserves the first 221 critic/Adam input rows, zero initializes rows 222--368
and retains Adam time `441976`.

V68 restores the complete 28-stage V37 order. It does not treat V67's two
isolated stages as the first two stages of this course and does not jump to
wide execution DR. V67 final to V68 Stage 1 changes exactly five environment
fields: the three physical DR correlation flags change from independent
per-joint to the V37 global episode-draw contract, and the RMP/complete-DR
privileged critic flags turn on. Interval endpoints remain half-spin and the
temporary V67 10% hard-reset-key mixture is removed.

```text
1        recover under the V37 small-execution physical-correlation contract
2--19    target hold, geometry, observation, camera and wide task support
20       complete non-execution polish (conv_len/full=0.95/0.90)
21--22   RMP-internal 6.25% execution DR + polish
23--24   complete 12.5% RMP-internal DR + polish
25--26   PD/plant 6.25% execution DR + polish
27--28   complete RMP/PD/plant execution DR + polish
```

Relative to V37, every stage changes only the three ball-spin intervals from
`+/-25,+/-25,+/-20` to V67's accepted candidate
`+/-12.5,+/-12.5,+/-10 rad/s`; rewards, observations, control, all other DR
endpoints and ordering remain V37-exact. Two survival gates alone are
moderately relaxed: Stage 1 uses `conv_len/full=0.93/0.92`, and Stage 20 uses
`0.95/0.90` instead of the statistically impractical `1.0/1.0` zero-failure
criterion over its 24-update stochastic-DR hold. Every conv_len gate remains
at least `0.90`; hit, apex, view, safety and every other gate are unchanged.
Stage 20 remains
`rmp68_complete_nonexecution_full_episode_polish`; only Stage 21 begins the
large execution-DR ladder. CMDP, transactional guards, outcome advantage
weighting and hard-key oversampling are prohibited.

The formal launcher uses GPU0, 1024 environments x 128 steps, minibatch
16384, three epochs, LR `1.0e-4`, clip `0.15`, target KL `0.008`, entropy
`0.0002`, block validation, no stage update cap, seed `20260940` and a new
offline W&B/output identity. The active output is
`outputs/rl_sim/measured_qvel_rmp_vertical_v68_gpu0_seed20260940_20260826_V67_final_complete_28stage_resume1`.
Startup reported `CudaDevice(id=0)` and the required actor-preserving,
Adam-preserving critic migration `57/221 -> 57/368`. At Stage-1 update 13 the
eligible rolling window was hits `14.578`, length fraction `0.9751` and full
`0.9525`; update KL was `0.00567`, qvel exceedance was zero and qacc
exceedance was `0.00447`. It subsequently graduated Stages 1--3. Stage 3
finished at update 95 with rolling hits `14.4103`, conv_len `0.95955` and full
`0.93820`. At the resulting Stage-4 entry, true step `2188640256` and global
update 385, the host-memory guard triggered `sustained_low_available` after
GPU1 had begun a concurrent 1024-environment V130 run. The trainer handled
SIGINT and saved identical interrupted/last checkpoints rather than crashing.

The exact safe-stop checkpoint has dimensions `57/368/7`, Adam time `449228`
and SHA-256
`c4a591d6ad09b583f5c8e40530a09672f1afef57b88bf180ca495766718573e9`.
Its causal 385-row progress CSV has SHA-256
`d9d6c80cbae35388e860fd3320ff0deadd7901d83db5aced31558e07784a7a76`.
Resume it only with
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v68_resume_interrupted_online.sh`.
The recovery starts Stage 4 with an empty local convergence window because the
checkpoint was written before its first rollout, but preserves global update,
actor, critic, optimizer and the completed-stage history. It uses a new output
directory, 1024 environments and W&B online run
`fushi37/pingpong-mjx/v68g0a1` with `resume=must`; no optimizer/critic reset,
observation migration, curriculum reinterpretation or W&B step offset is
allowed. The online recovery output is
`outputs/rl_sim/measured_qvel_rmp_vertical_v68_gpu0_seed20260940_20260826_stage4_interrupted_resume2_online`.
At Stage-4 update 17 its rolling hits/conv_len/full were
`14.4666/0.96280/0.94307`, KL was `0.00599`, qvel exceedance was zero and qacc
exceedance was `0.00418`. W&B state was `running` and remote history had
reached step `2190606336`, within two updates of local logging. V68 remains
experimental, simulation-only and inherits every
unresolved camera, replacement-ball and formal RMP/PD evidence blocker.

## Strict Stage-7 Restart V69

Status (2026-08-26): active on GPU0 with 1024 environments and W&B online.

Profile `goal_d455_measured_qvel_rmp_vertical_v69` launches only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v69_restart_stage7_strict.sh`.
It pins the V68 Stage-6 completion checkpoint
`06_rmp68_nonexecution_launch14_racket_geometry_wide.pkl`, true step
`2250244096`, Adam time `460120`, dimensions `57/368/7`, SHA-256
`b82411bb6e19546fa2396c21112b90c2b2d3e3241b8c68a81719620f6bbf4513`.
The source finished a 12-update strict hold with rolling
hits/conv_len/full `14.4150/0.96108/0.93032`. It is the last accepted boundary
before V68 Stage-7 optimization. V69 restores actor, critic and Adam exactly,
starts fresh Stage-7 convergence history and prohibits curriculum-history
resume, observation migration, parameter/optimizer reset and CMDP.

The restart is required because V68 Stage 7's compensatory `balanced_probe`
gate accumulated its 12-update hold and advanced with rolling
hits/conv_len/full `12.758/0.87214/0.77906`, below its strict numerical targets
`13.0/0.9025/0.82`. At V68 Stage-8 update 305 the same three metrics had
degraded to `10.476/0.74008/0.55959`; the next-domain block probe also failed
mean hits (`4.298 < 4.55`) before the host-memory guard safely stopped the
process. The Stage-7 and Stage-8 checkpoints are negative evidence and must
not seed later lessons.

V69 is a one-to-one copy of all 28 V68 stages. Environment, reward, control,
reset, DR, target values, minimum updates, hold lengths and block-validation
settings are identical. The only semantic change is that Stages 7--10 use
`gate_mode=strict` instead of `balanced_probe`. All current-domain metrics
must therefore pass independently throughout the hold. Their
`advance_gate_mode=collapse` next-domain probes remain unchanged because those
probes test transfer into the deliberately harder following lesson. Every
course `conv_len` target stays at or above `0.90`; Stage 20 remains the
complete non-execution polish and wide execution DR still begins only at
Stage 21.

The launcher keeps 1024 environments x 128 steps, minibatch 16384, three PPO
epochs, LR `1.0e-4`, clip `0.15`, target KL `0.008`, block validation and no
stage update cap. It uses W&B online with a new identity rather than
reinterpreting V68 history. V69 is experimental and simulation-only and
inherits every unresolved camera, replacement-ball and formal RMP/PD evidence
blocker.

The active output is
`outputs/rl_sim/measured_qvel_rmp_vertical_v69_gpu0_seed20260940_20260826_restart_stage7_strict_pp_gpu0_online1`;
it was launched from tmux pane `pp_gpu0:0.0`, and W&B run ID is `v69g0s7b1`.
Startup reported `CudaDevice(id=0)`, restored true
step `2250244096`, entered Stage 7/28 with dimensions `57/368/7`, and completed
its first update at true step `2250375168`.

V69 was safely stopped at Stage-7 update 1257, true step `2415001600`, after
the low-orbit failure became conclusive. Final rolling hits/conv_len/full were
`12.8388/0.89022/0.81036`; absolute apex/lift/contact ball z were
`1.27654/0.16032/1.11623 m`. The interrupted and last checkpoints are
identical, SHA-256
`00f914db575ef2c0bd9c5c0daa438b2576fe991fe8b4d82c67f8d551e1d814c5`.
Neither checkpoint is an eligible successor source.

## Height/Survival-Aligned Stage-7 Restart V70

Status (2026-08-26): implemented for an exact-shape bounded GPU0 trial; a
formal no-cap run requires the preregistered trial to pass.

Profile `goal_d455_measured_qvel_rmp_vertical_v70` launches only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v70_restart_stage7_reward_aligned.sh`.
It restarts the exact clean V68 Stage-6 completion archive at true step
`2250244096`, Adam time `460120`, SHA-256
`b82411bb6e19546fa2396c21112b90c2b2d3e3241b8c68a81719620f6bbf4513`.
This source completed its strict hold at hits/conv_len/full
`14.4150/0.96108/0.93032` and precedes all V69 Stage-7 optimization. V70 must
reject V69, another profile/step/hash, curriculum-state resume, parameter or
optimizer reset, observation migration, CMDP and advantage reweighting.

V70 copies all 28 V69 stages and preserves every environment, control,
observation, reset, physics, DR interval, gate and next-domain block probe.
Stages 1--6 remain config-identical and are not replayed. Stage 7 onward
changes only this aligned reward group:

| field | V69 | V70 |
| --- | ---: | ---: |
| absolute apex target / sigma / event weight | 1.32 / 0.050 / 0.75 | 1.30 / 0.035 / 1.5 |
| absolute lower edge / weight | 1.26 / 0 | 1.28 / 7000 |
| lift target / sigma / event weight | 0.175 / 0.040 / 0.75 | 0.175 / 0.030 / 1.0 |
| lift barrier / weight | 0.130--0.220 / 100 | 0.145--0.205 / 500 |
| post-hit joint apex/lift | off | weight 8 for 60 ms |
| full 1200-step joint height | off | weight 20, only 13--15 hits |
| upward-racket drift penalty | 6 | 0 |

Dense post-first-hit alive weight 5, post-hit recoverability weight 2.4, the
14-event positive-hit cap, contact-edge ball/racket height barriers, view,
motion and safety rewards remain exact. The absolute lower edge now has a
real PPO gradient instead of existing only as a curriculum gate. The short
joint term improves action-local credit, while the terminal joint term can be
earned only by simultaneously completing 1200 steps with 13--15 hits; this is
the explicit mechanism preventing height from being optimized by giving up
survival.

GPU1 V130's lower update scale is used only as optimizer evidence, not as an
action/control transfer. V70 uses 1024 environments x 128 steps, two epochs,
LR `5e-5`, clip `0.10`, target KL `0.004`, entropy `0.0002`, min log standard
deviation `-4.8`, a 48-update convergence window and 16-update warmup.
Stages 7--10 retain strict gates and require a 24-update hold. GPU1's QACC
residual L2 and actor-anchor terms are not copied into this QVEL/RMP policy.

The first launch sets `V70_BOUNDED_TRIAL=YES`, uses the exact formal batch
shape for 64 updates, stops after Stage 7 without forced advancement, logs
W&B offline and writes a unique directory. Acceptance requires a final
window with hits 13--15, conv_len >=0.90, full >=0.82, absolute apex >=1.285 m
with no early-to-late decline, lift >=0.160 m, qvel exceedance zero, qacc
exceedance <=0.01 and mean exact KL <=0.004. A higher return is not acceptance.
Only a passing trial may launch the no-cap online configuration with a new
W&B ID and output directory. V70 remains experimental, simulation-only and
inherits every unresolved camera, replacement-ball and formal RMP/PD blocker.

## Coupled Height/Survival Stage-7 Restart V71

Status (2026-08-26): rejected by its exact-shape bounded GPU0 trial; no formal
online run is permitted.

The V70 64-update trial is a preserved negative result. Its final 48-update
window measured hits/conv_len/full `12.9638/0.88073/0.79077`, absolute
apex/lift `1.29160/0.16572 m`, mean exact KL `0.00283`, qvel exceedance zero
and qacc exceedance `0.00425`. The new height objective fixed the low-orbit
drift, but survival did not reach the registered `0.90/0.82` thresholds.
Racket-too-high terminations averaged about `12.75` events per 131072-sample
update, versus `8.27` at the late V69 window; fully removing the upward-drift
guard was therefore not accepted. V70 is not eligible for formal launch or as
a successor checkpoint.

Profile `goal_d455_measured_qvel_rmp_vertical_v71` launches only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v71_restart_stage7_coupled_survival.sh`.
It restarts the exact V68 Stage-6 boundary at true step `2250244096`, Adam
time `460120` and SHA-256
`b82411bb6e19546fa2396c21112b90c2b2d3e3241b8c68a81719620f6bbf4513`.
Every V70 course, DR, gate, environment and optimizer value is preserved.
From Stage 7 onward only two reward weights change:

| field | V70 | V71 |
| --- | ---: | ---: |
| complete 1200-step, 13--15-hit joint apex/lift reward | 20 | 40 |
| upward-racket drift penalty | 0 | 2 |

The joint completion reward still requires both the absolute-apex and lift
targets, so it cannot reward a low orbit or an incomplete episode. The weak
drift guard is one third of V69's weight 6 and is paired with V70's active
`1.28 m` lower barrier; it targets racket-high termination without restoring
the old incentive to lower the ball. Dense alive/recoverability weights stay
`5/2.4`, and all view, contact-height and safety terms are exact.

The trial's final 48-update hits/conv_len/full were
`12.8766/0.87763/0.77721`, while apex/lift remained
`1.29112/0.16589 m`. Racket-too-high terminations averaged `12.79` per
131072-sample update, effectively unchanged from V70's `12.75`. The coupled
reward increased the value-target scale but did not improve survival. V71 is
not eligible for formal launch or as a successor checkpoint.

## Stable Observation-DR-Cap Stage-7 Restart V72

Status (2026-08-26): exact-shape bounded GPU0 acceptance passed. The first
no-cap W&B-online process safely stopped at Stage 7 update 248 under its
GPU0-only host-memory guard; its exact-state recovery continues through the
registered resume launcher in `pp_gpu0`.

Four independent histories now reproduce the same Stage-7 boundary failure:
the V68 original tail, the long V69 restart, and the V70/V71 reward-repair
trials all plateau near conv_len `0.88` after the original 25% in-view
observation-calibration DR is enabled. The V68 Stage-6 source completed at
hits/conv_len/full `14.4150/0.96108/0.93032`. V70/V71 kept physical apex near
`1.291 m`, so the remaining gap is a reproducible observation-domain problem,
not the prior height-reward shortcut or unstable PPO.

Profile `goal_d455_measured_qvel_rmp_vertical_v72` launches only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v72_restart_stage7_obsdr_cap.sh`.
It restarts the exact V68 Stage-6 boundary, preserves V70's complete reward,
28-stage ordering, Stage-20 complete polish, gates, PPO, control, resets,
physics, target support and every non-observation DR field. From Stage 7
onward only these six unsupported in-view calibration fields are capped at
the exact 12.5% midpoint between V70 Stages 6 and 7:

| field | V72 cap |
| --- | ---: |
| ball position noise std | `0.004375 m` |
| ball velocity noise std | `0.04375 m/s` |
| coherent position bias | `[0.0025,0.0025,0.0025] m` |
| coherent rotation bias | `[0.43125,0.43125,0.625] deg` |
| coherent velocity bias | `[0.025,0.025,0.03625] m/s` |
| observation scale | `[0.996125,1.003875]` |

This is a stable candidate bound, not real-camera coverage evidence. The
25%--100% range remains unsupported by a componentwise hardware audit and is
not retained merely for nominal width when it reproducibly violates the
minimum `0.90` conv_len contract. V72 keeps the same 64-update acceptance as
V70 and does not lower any gate.

To protect the concurrently running GPU1 job, the V72 launcher pins a
GPU0-only early-stop guard: 6.25 GiB sustained / 5 GiB critical available RAM,
256 MiB launch-relative swap growth under 8.5 GiB available RAM, one-check
grace, and 512 MiB swap reserve. It can signal only its own GPU0 child and is
intentionally stricter than the shared default guard. GPU1 is read-only and
must never be signalled or reconfigured by this workflow.

The accepted run is
`outputs/rl_sim/measured_qvel_rmp_vertical_v72_gpu0_seed20260943_20260826_stage7_obsdr_cap_trial64b`
(offline W&B ID `v72g0s7trialb`, physical GPU0 UUID
`GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`). Its final 48-update window was:

```text
hits / conv_len / full             13.69995 / 0.92047 / 0.85965
hit1 / hit3 / hit12                0.99304 / 0.95846 / 0.86234
absolute apex / lift               1.28948 / 0.16540 m
counted adjacent-hit interval      0.36780 s
true view / camera visible         0.97903 / 0.99799
mean / RMS ball vxy                0.12673 / 0.15981 m/s
mean / RMS hit-racket vxy          0.09792 / 0.11488 m/s
full racket angular speed          1.25818 rad/s
ball / racket contact z            1.12409 / 1.10434 m
qvel / qacc exceedance             0 / 0.00403
mean exact KL / explained variance 0.00300 / 0.89595
```

The first-16 versus last-16 absolute apex was `1.28742 -> 1.28980 m`; there
was no height-for-survival collapse. Against matched V70 Stage 7, V72 also
improved true view by `0.01004`, lowered contact heights, reduced ball and
racket lateral speed, reduced qacc exceedance, and reduced racket-high
terminations from `12.75` to `8.02` per update. The 64-update cap ended before
the unchanged 225-update stage minimum, as preregistered; this is not forced
advancement. Never resume the bounded checkpoint. Formal training restarts
the pinned V68 Stage-6 optimizer source in a new output/W&B identity.

The active formal identity is
`outputs/rl_sim/measured_qvel_rmp_vertical_v72_gpu0_seed20260943_20260826_stage7_obsdr_cap_online1`,
W&B `v72g0s7a1`. Startup confirmed online sync, JAX `CudaDevice(id=0)`,
physical GPU0 UUID `GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`, Stage 7/28,
and no per-stage update cap. The run completed initial updates with the
independent GPU1 process still alive. GPU1 remains read-only; monitoring or
continuing V72 is not permission to signal or reconfigure it.

The safe-stop checkpoint is `mjx_curriculum_interrupted.pkl` at true step
`2282749952`, Stage 7 local/global update `248/248`, Adam `t=464071`, SHA-256
`f1990c963a4c98f109a654bb5cb5f7e9b675c28d92023547a91813d601f07891`.
The source progress CSV has SHA-256
`4e454abb0c67a6c53f49ebddb46f1b374731d5f4b5f28bd24bc42947b6ceb6c9`.
The stop cause was `sustained_low_available`, not PPO, CUDA, temperature or
checkpoint corruption. At the last 48-update window, hits/conv_len/full were
`13.6475/0.92/0.86`, apex/lift were `1.294/0.168 m`, and the strict current-
domain gate was passing.

Resume only with
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v72_resume_interrupted_online.sh`.
It creates
`outputs/rl_sim/measured_qvel_rmp_vertical_v72_gpu0_seed20260943_20260826_stage7_obsdr_cap_online1_resume1`,
restores actor, critic, Adam, stage anchor and all causal Stage-7 history, and
continues W&B `v72g0s7a1` with `resume=must` and offset zero. It retains 1024
environments and every V72 PPO/course value. The self-resume validator and
launcher hashes fail closed against a different checkpoint, profile, stage,
domain, dimension, optimizer, progress history or non-signal stop. GPU1 is
strictly read-only and only the GPU0 child may be stopped by this launcher.

## GPU0 V73 Spinless Stage-9 Continuation

Status (2026-08-26): registered and validated for formal GPU0 continuation.
The V72 recovery completed Stage 8 and entered one-based Stage 9,
`rmp72_nonexecution_launch17_observation_calibration_wide`, at local update
zero before its GPU0-only host-memory guard safely requested SIGINT. The exact
source is
`outputs/rl_sim/measured_qvel_rmp_vertical_v72_gpu0_seed20260943_20260826_stage7_obsdr_cap_online1_resume1/mjx_curriculum_interrupted.pkl`
at true step `2326003712`, global update `578`, Adam time `468247`, SHA-256
`365ab6ef01e7fc79b7f3102f3263ba684b44a6b4ea10fef166191da723392832`.
Its progress CSV SHA-256 is
`a19fe50b0969a8fe600c8bbf9619a5cd1369f28b55540fd48aac80053329e498`.
The last causal V72 Stage-8 window had hits/conv_len/full
`13.60808/0.91667/0.85846` and absolute apex/lift
`1.29399/0.16723 m`; Stage 8 advanced normally.

Profile `goal_d455_measured_qvel_rmp_vertical_v73` launches only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v73_zero_spin_resume_v72.sh`.
It is a one-to-one copy of all 28 V72 stages and preserves Stage 20 as complete
non-execution polish. Every config, reward, gate, observation, reset, control,
PPO and non-spin DR value remains exact except these four spin-related fields
on every stage:

| field | V72 | V73 |
| --- | ---: | ---: |
| reset spin X | `[-12.5,12.5] rad/s` | `[0,0] rad/s` |
| reset spin Y | `[-12.5,12.5] rad/s` | `[0,0] rad/s` |
| reset spin Z | `[-10,10] rad/s` | `[0,0] rad/s` |
| normalized inertia | `[0.54,0.6666667]` | `[0.40,0.40]` |

This matches GPU1 V131's spinless interpretation. Natural angular velocity
generated by racket/ball contact remains enabled in MuJoCo; V73 removes reset-
spin and normalized-inertia randomization, not rotational dynamics. It is an
experimental domain choice, not evidence that real ball spin is covered.

V73 restores the 57-D actor, 368-D critic and all Adam moments but starts a
fresh Stage-9 convergence window and stage-local actor anchor because the DR
semantics change. Curriculum-history resume, optimizer/critic reset,
observation migration, another source checkpoint and from-scratch launch are
prohibited. The formal launcher keeps 1024 environments by 128 steps,
minibatch 16384, two epochs, LR `5e-5`, clip `0.10`, target KL `0.004`, the
48-update window, 16-update warmup, no per-stage update cap and the V72 GPU0-
only host-memory guard. By explicit user authorization it continues W&B
online run `v72g0s7a1` with `resume=must` and offset zero while writing a new
V73 output directory. GPU1 is strictly read-only.

The formal run was launched in `pp_gpu0` on 2026-08-26 as
`outputs/rl_sim/measured_qvel_rmp_vertical_v73_gpu0_seed20260943_20260826_stage9_zero_spin_online1`.
Startup confirmed W&B-online `resume=must` for run `v72g0s7a1`, physical GPU0
through JAX `CudaDevice(id=0)`, source true step `2326003712`, and V73 Stage
9/28. The run completed at least eight updates, reaching true step
`2327052288` at about `19.7k steps/s`; its progress row reports normalized
inertia `0.400000...` and zero reset-spin draws. GPU1 remained running and was
not signalled or reconfigured.

The same run subsequently completed Stage-9/global update 263 at true step
`2360475648` and then stopped safely because the GPU0-only host-memory guard
detected `swap_growth_under_pressure`. Its interrupted checkpoint SHA-256 is
`020b40a1bdf9cbd00d163084349cfaa52146b9744d174f4eb8a0f04499bdef53`;
the causal progress CSV SHA-256 is
`914726f4c1bd73631224d49f84bbeae866814559f0eb56cc962d14225c094e57`.
The final rolling hits/conv_len/full were
`13.63216/0.92138/0.85545`; absolute apex/lift were
`1.29231/0.16747 m`. Every Stage-9 gate passed, the strict hold reached 24/24,
and the Stage-10 block validation passed. The signal arrived before the
trainer committed the already-proven transition, so repeating Stage 9 is not
required.

The fail-closed recovery launcher is
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v73_resume_interrupted_online.sh`.
It pins both hashes, verifies the causal convergence and next-stage probe,
restores the 57-D actor, 368-D critic and Adam time `472455`, and enters exact
Stage 10 `rmp73_nonexecution_launch18_camera_missing_wide` with a fresh
convergence window and actor anchor. It preserves 1024 environments and all
V73 PPO/course settings, writes
`outputs/rl_sim/measured_qvel_rmp_vertical_v73_gpu0_seed20260943_20260826_stage10_zero_spin_online1_resume1`,
and resumes the same W&B run `v72g0s7a1` with `resume=must` and zero offset.
Startup on 2026-08-26 confirmed JAX `CudaDevice(id=0)`, source true step
`2360475648`, and Stage 10/28. It completed at least 22 1024-by-128 updates,
reaching true step `2363359232` at about `19.6k steps/s`; rolling gate metrics
became available normally after the intended 16-update warmup. GPU1 is
strictly read-only.

## GPU0 V74 Negative Trial and V75 Camera-Missing Recovery

Status (2026-08-26): V74 retained as a negative bounded experiment; V75
accepted for formal GPU0 continuation. The V73 Stage-10 run stopped safely at
update `491`, true step `2424832000`, because its GPU0-only host-memory guard
detected `swap_growth_under_pressure`. The immutable interrupted/last source
checkpoint has Adam `t=478857`, SHA-256
`9dca0f250f30a6c53aa33baf9f61a0a690607d3713ac09cf372f359717c8c559`;
the progress CSV SHA-256 is
`47bf4cf27dc3d4357738679aa11af60cfde75693fa7261ece41d331efe35f39a`.

Full trajectory analysis found no PPO explosion or abnormal height-policy
collapse. V73 Stage 9 to Stage 10 simultaneously changed camera/view missing
`0.15 -> 0.50`, refresh dropout `0.004 x 1 -> 0.012 x 3`, conv_len
`0.91 -> 0.93`, and next-contact recovery `0.1305 -> 0.126 m`. The long-run
Stage-10 slopes were slightly negative for hits/length/full while height,
view, cadence and safety remained physical. Paired frozen screens rejected
rollback to the saved update-111 best because update 491 was more robust over
the 15%, 25% and 35% conditions.

V74 inserts 25% and 35% observation-missing bridges before the exact V73 50%
domain and moves complete non-execution polish from Stage 20 to Stage 22. Its
128-env, 32-update bounded trial was negative under the preregistered criteria:
last-12 weighted hits/conv_len/full were `13.2211/0.90171/0.81053`, missing
the required `0.91/0.82` longevity levels despite positive slopes, physical
apex/lift, exact KL `0.003447`, zero qvel exceedance and qacc exceedance
`0.004289`. V74 remains immutable and is not the formal successor.

Profile `goal_d455_measured_qvel_rmp_vertical_v75` changes only the rejected
first V74 bridge to 20% camera/view missing with `0.005 x 1` refresh dropout.
The later 35% and exact V73 50% stages, every reward/control/physics/reset/spin
and non-observation DR field, PPO settings, and all later stages remain exact.
Its 128-env, 32-update trial passed: last-12 weighted hits/conv_len/full were
`13.5847/0.91514/0.83060`; first-six to last-six length rose
`1077.80 -> 1120.64`, full rate `0.8021 -> 0.8621`, and hits
`13.2396 -> 13.9655`. Apex/lift were `1.29672/0.16948 m`, exact KL
`0.003636`, qvel exceedance zero and qacc exceedance `0.004369`.

V75 is continuation-only and launches through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v75_resume_v73.sh`.
It must restore the exact V73 update-491 57-D actor, 368-D critic and Adam
state, prohibit curriculum history and optimizer/critic reset, and start a
fresh V75 Stage-10 bridge window. Formal training fixes 1024 environments,
128 steps, minibatch 16384, two epochs, LR `5e-5`, clip `0.10`, target KL
`0.004`, a 48-update window and 16-update warmup. By explicit user
authorization it continues W&B `v72g0s7a1` with `resume=must` and offset zero.
The GPU0-only memory guard may signal only its own child; GPU1 is read-only.

The initial formal V75 compile exposed that two simultaneous 1024-environment
trainers do not fit the 32 GiB host-memory envelope: GPU0's polling guard and
GPU1's independent guard both safely stopped their own children before V75
completed an update. The retry must therefore launch through
`run_gpu0_measured_qvel_rmp_vertical_v75_cgroup_resume_v73.sh`, which adds a
GPU0-process-tree-only cgroup with `MemoryHigh=9 GiB`, `MemoryMax=9 GiB` and
`MemorySwapMax=0` outside the existing polling guard. It writes a new output
directory and reuses the immutable V73 source because the aborted V75 payload
was still at stage/global update zero and W&B uploaded no training-history
point. This resource wrapper changes no PPO or curriculum semantics.
The first isolated 7 GiB probe reached the exact hard limit and OOM-killed
only its GPU0 scope before update zero completed; the registered 9 GiB value
adds the measured compile headroom while retaining zero scope swap.

The 9 GiB formal V75 retry wrote
`outputs/rl_sim/measured_qvel_rmp_vertical_v75_gpu0_seed20260943_20260826_camera_missing_bridge_online1_resume2`
and completed 74 1024-by-128 PPO updates. Its final complete 48-update window
had hits `13.5103`, conv_len about `0.916`, full rate `0.84`, next-contact
error `0.134 m`, absolute apex `1.297 m` and lift `0.170 m`; the complete
strict gate remained passing. This disproves a continuing policy-regression
failure at the 20% bridge. The scope then OOM-killed GPU0 while compiling the
128-environment block validation: cgroup peak was exactly 9 GiB, scope swap
remained zero, and `memory.events` recorded `oom_kill=2`. The last periodic
checkpoint is Stage-10/global update 70, true step `2434007040`, Adam
`t=479945`, SHA-256
`de7ea8b75a7125ffe5dec4227e91a167af8bde8a2646d52eb9aad2d529a7cd1c`.
The causally filtered progress CSV SHA-256 is
`18d5bc343421a8fae8e18f804b8d9e2af572ff524664c17b83d7eedc4fa5ccc0`.

Profile `goal_d455_measured_qvel_rmp_vertical_v76` is the resource-only exact
continuation. It freezes all 30 V75 stages byte-for-byte, including their V75
stage identities, and must restore the pinned update-70 actor, critic, Adam,
actor anchor and convergence history. It changes no reward, DR, gate, PPO or
1024-environment training field. Only block-validation parallelism changes
from 128 to 64 environments; the unchanged validation horizon and minimum 64
episodes retain one complete validation episode per lane. The W&B-only offset
is `524288`, placing its first resumed point after V75's unsaved update-74
history without changing true optimizer/global steps. Launch only through
`run_gpu0_measured_qvel_rmp_vertical_v76_cgroup_resume_v75.sh`, whose complete
GPU0 process tree has `MemoryHigh=10 GiB`, `MemoryMax=10 GiB` and
`MemorySwapMax=0`. The launcher never addresses GPU1.

V76 completed Stage 10 at update 135. Its 64-episode block validation passed
with hits `13.90625`, length fraction `0.93954` and full-episode rate `0.875`,
so the trainer correctly advanced to Stage 11. The stage-end checkpoint is at
true step `2442526720`, Adam `t=480793`, SHA-256
`3a33791a31bff790c8c6c95cc309954639b048c2f5466aa6cde53f098e9a5c6d`;
the progress CSV SHA-256 is
`01882073e7be2c23f2f3ade97970528f521f35797be7fead54497e96818c0fa7`.
V76 then OOM-killed while compiling Stage 11. The scope reached its exact
10 GiB `MemoryMax` and recorded `oom_kill=3`, even though the host had about
24 GiB available and GPU1 had no compute process. Stage-10 training occupied
about 9.0 GiB and its validation raised the retained process footprint to
about 9.88 GiB. This proves that lowering validation lanes delayed but did not
fix the local hard-cap plus cross-stage JAX/XLA cache problem.

Profile `goal_d455_measured_qvel_rmp_vertical_v77` is the OOM-repair
continuation. It freezes every V75/V76 course, reward, gate, DR, reset,
control and PPO field and resumes only the V76 Stage-10 exit at exact Stage 11
`rmp75_nonexecution_camera_missing_bridge_35`. It restores formal 1024-env
training and 128-env block validation; lowering either is prohibited. The
trainer's new opt-in `--clear-jax-caches-between-stages` first saves the
completed stage, drops strong references to that stage's environment,
rollout/update closures and arrays, then calls `jax.clear_caches()` and Python
garbage collection before constructing the next stage. The switch defaults
off, so historical launchers retain their original behavior.

Launch V77 only through
`run_gpu0_measured_qvel_rmp_vertical_v77_cgroup_resume_v76.sh`. Its GPU0-only
scope uses `MemoryHigh=11 GiB`, `MemoryMax=12 GiB` and `MemorySwapMax=0`:
the hard limit is raised above the measured train/validation/compile peak
while retaining a bounded process tree and enough worst-case space for the
other measured 9--10 GiB 1024-env trainer on this 32 GiB host. Its host guard
uses the repository's dual-trainer envelope: a 4 GiB sustained-available
warning, 2 GiB emergency line and up to 4 GiB launch-relative swap growth,
while cgroup swap remains disabled for GPU0. It starts a fresh
Stage-11 convergence window, continues W&B `v72g0s7a1` with `resume=must` and
the unchanged W&B-only offset `524288`, and never signals or configures GPU1.
The 12 GiB bound and cache release are resource fixes, not permission to
change the 1024/128 experiment contract.

Detailed preregistration, frozen screens, negative result and accepted V75
decision are under
`outputs/rl_sim/v74_stage10_camera_missing_bridge_20260826/`.

## GPU0 V78 Stage-13 Noisy-Observation Survival Repair

Status (2026-08-27): bounded validation accepted for formal GPU0 continuation. V77
was stopped safely after Stage-13 update `9194` / global update `9431`, true
step `3678666752`; W&B run `v72g0s7a1` synced successfully. The interrupted
checkpoint SHA-256 is
`9a9c673928835750ccf46c42f95a6e5deb1bea0fc03abf8393f0e7667c698fca`.
V77 did not have PPO/KL/value divergence: over more than 8000 Stage-13
updates, `conv_len` and full-episode rate oscillated near `0.927/0.863`, while
full racket angular speed remained near `1.48 rad/s` and next-contact anchor
error near `0.13 m`. The original strict joint gate was therefore not a
single survival threshold; its angular and anchor requirements had almost no
continuous overlap with the survival window.

Frozen first-episode-per-lane screens compared V77 Stage-13 updates 600, 750,
800, 5000, the legacy `best`, and the interrupted policy without resampling
fast failures. The selected parent is update 800, true step `2578448384`,
global update `1037`, Adam time `497258`, SHA-256
`c20a4b9178089f007aa8a9cd3401e56cd18484ee775786b3f4066dc517822629`.
Across two 128-lane seeds it produced hits `13.879`, `conv_len=0.9412` and
full rate `0.8867`; on the quality-instrumented seed it had full racket
angular speed `1.448 rad/s`, hit-racket VXY `0.096 m/s`, next-contact anchor
error `0.124 m`, zero qvel exceedance and qacc exceedance `0.00400`. It was
selected from this Pareto evidence, not return or the legacy best score.

Paired causal replay shows the residual failure population is real but not a
single unreachable DR corner. Ball/contact DR nominalization moved full rate
only about `0.883 -> 0.898`; RMP/PD nominalization moved it to `0.945`, and
observation/mount nominalization to `0.961`. Disabling only refresh-sample
ball-position/velocity noise, while retaining dropout, view-missing sampling,
target holds and all episode DR, completed all `128/128` episodes in the
registered two-seed screen. Disabling noise plus dropout gave `127/128` full
episodes. Target-hold-only results were mixed across paired lanes. Thus the
dominant tail is the 57-D feed-forward actor's sensitivity to per-refresh
ball-state noise, amplified by execution/observation DR; it is not exploration
noise or ball contact alone. These candidate noise bounds still lack the
paired real-camera audit required by the V16 contract and must not be called
hardware coverage.

Failure traces show the correct reward target is a local boundary precursor,
not a global height drift penalty. In failed trajectories, sampled
`racket_z_rel` exceeded `0.12 m` in `8.3%` of samples versus `1.9%` for full
episodes; during the last 100 steps its 90/95/99 percentiles were about
`0.196/0.215/0.238 m` against the unchanged `+0.24 m` hard limit. The old
soft term used unnormalised squared metres, weight 15 and a `+0.14/0.00 m`
band, so it supplied little pre-terminal gradient. V78 changes only this
reward family from its entry stage onward:

```text
racket_z_band_up:               0.14 -> 0.12 m
racket_z_band_down:             0.00 -> 0.04 m
racket_z_soft_penalty_weight:  15.0 -> 60.0
racket_z hard limits:          unchanged +0.24 / -0.12 m
racket_up_drift penalty:       remains zero
```

Frozen trace re-scoring estimates the new integrated penalty at only about
`-0.010` per full episode versus `-0.060` per failed episode; it leaves the
healthy central band effectively untouched and does not recreate V69's global
low-orbit shortcut. Every observation/noise/dropout, reset, physics, RMP/PD,
contact, action, apex/lift, survival and terminal-reward field otherwise
remains exact V77.

V78 inserts two unchanged-domain lessons before V77's later observation tail.
The survival bridge uses `conv_len>=0.92`, full rate `>=0.85`, full racket
angular speed `<=1.45 rad/s` and next-contact anchor error `<=0.130 m`. The
quality commit keeps `0.92/0.85` but restores angular `<=1.40 rad/s` and
tightens anchor error to `<=0.125 m`. No stage may use `conv_len<0.90`; all
later V77 stages remain in order with their original gates, making complete
non-execution polish V78 Stage 23 and leaving execution DR strictly after it.
Best-checkpoint ranking is opt-in corrected from the repair stage onward to
include full racket angular speed and next-contact anchor error; historical
profile rankings are unchanged.

V78 is continuation-only. It restores the pinned V77 update-800 actor,
critic, Adam moments and stage actor anchor, starts fresh repair convergence
history, and rejects curriculum-history resume, reset flags, CMDP,
transaction guards, advantage reweighting and observation migration. Bounded
training may use 512 environments/minibatch 8192; the formal contract remains
1024 environments/minibatch 16384, rollout 128, two epochs, LR `5e-5`, clip
`0.10`, target KL `0.004`, a 48-update window, 16-update warmup, stochastic
128-environment block validation and no stage cap.

Launch only through
`run_gpu0_measured_qvel_rmp_vertical_v78_cgroup_resume_v77.sh`. Its GPU0-only
scope retains `MemoryHigh=11 GiB`, `MemoryMax=12 GiB` and `MemorySwapMax=0`;
GPU1 remains outside the mutation scope. The default formal launcher resumes
W&B `v72g0s7a1` with `resume=must` and W&B-only offset `1100742656`, placing
the first V78 point exactly one rollout after the V77 tail without changing
the older checkpoint's true optimizer/global step. Bounded trials must use a
new output directory and a separate offline W&B identity.

The accepted bounded run is
`measured_qvel_rmp_vertical_v78_gpu0_seed20260943_20260827_stage13_boundary_barrier_trial64b`
(512 environments, 64 updates, offline W&B `v78g0s13barriertrial2`). Its final
rolling `hits/conv_len/full` were `13.716/0.9340/0.8773`; angular speed was
`1.413 rad/s`, hit-racket VXY `0.0933 m/s`, absolute apex/lift
`1.2939/0.1684 m`, qvel exceedance zero, qacc exceedance `0.00402`, and
tail-24 KL `0.00272`. Across two first-episode-per-lane 128-lane seeds, V77
update 800 versus V78 update 50 versus V78 last had pooled full rates
`0.8789/0.9063/0.9023`, mean lengths `1131.4/1137.2/1138.7`, full angular
speeds `1.455/1.431/1.430 rad/s`, and hit-racket VXY
`0.0994/0.0978/0.0971 m/s`. Racket-high terminals fell from 15 to 11/12;
qacc behavior was unchanged. The trial's update-50 and last SHA-256 values
are `1d0e516a77bd0c0c09f326b81ed653b817d456e2563607a2a886be954d3c39c7`
and `d0f562f264ac67e17eb81ec55a1769988319dfd752a309d84ca6a229e4916669`.
These 512-environment states remain bounded evidence; formal V78 deliberately
restarts the pinned V77 update-800 parent under the 1024-environment contract.
The full root-cause, counterfactual, failure-trace and validation record is in
`outputs/rl_sim/v78_stage13_survival_repair_20260827/ROOT_CAUSE_AND_BOUNDED_TRIAL_REPORT.md`.

Formal V78 started in tmux `pp_gpu0` at 2026-08-27 19:57 Asia/Shanghai under
the required cgroup. The output directory is
`measured_qvel_rmp_vertical_v78_gpu0_seed20260943_20260827_stage13_survival_repair_online1`.
JAX reported `CudaDevice(id=0)` on the pinned GPU0 UUID, W&B reported
`Resuming run gpu0-v78-stage13-survival-repair` for `v72g0s7a1`, and the
trainer restored the exact V77 update-800 state at true step `2578448384`.
At the recorded handoff it had completed six 1024-environment updates at
about 19.7--20.2k steps/s and reached true step `2579234816`. GPU1 remained
outside the command and signal scope.

## GPU0 V79 Angular/Anchor Reward-Alignment Repair

Status (2026-08-28): V79 is retained as negative sparse-event evidence.  V80
passed its bounded and two-seed frozen-validation contract and is the selected
formal successor; the exact decision evidence is recorded below.

Formal V78 later reached Stage 14
`rmp78_nonexecution_launch19_survival_quality_commit` but remained there for
more than 2000 updates.  Angular speed never passed its `1.40 rad/s` gate
(best rolling value about `1.4313`), while next-contact anchor error passed
early and then regressed from about `0.120` to `0.134`; the long-run
`conv_len/full` frontier remained about `0.92/0.86`.  The Stage-13-to-14
transition changed gates only, not rewards or domain, so it could reject a
policy but supplied no new quality gradient.  PPO remained stable (late KL
about `0.0026`, explained variance about `0.93`, qvel exceedance zero and qacc
exceedance about `0.0046`).  This is a reward/gate alignment stall, not an
optimizer divergence or evidence to widen/narrow DR.

V79 resumes only V78 Stage-14's quality-aware best at update 64, global update
143, true step `2597191680`, Adam time `499034`, SHA-256
`b481cd6aec2601696c2255f71ede4c18f3a1d9e4ca77c5f923092671e139c3db`.
Its rolling hits/length/full were `13.6926/0.93551/0.87463`, angular speed
`1.43585 rad/s`, anchor error `0.11828 m`, apex/lift
`1.29439/0.16995 m`, qvel exceedance zero and qacc exceedance `0.00401`.
The external stop did not produce a new transactional checkpoint, so neither
the degraded periodic last nor an inferred in-memory tail is an eligible
parent.  The validator pins profile, stage, dimensions, domain, PPO contract,
step, Adam time and stage actor anchor and rejects curriculum-history resume,
reset flags, CMDP and observation migration.

V79 freezes every V78 physics, DR, observation, reset, bounded-qref/RMP/PD,
PPO, height, survival and terminal-reward value.  It replaces V78 Stage 14 by
four unchanged-domain lessons:

```text
Stage 14 angular isolation   angular penalty 0.25 -> 0.35; anchor stays 0.03
Stage 15 anchor isolation    anchor penalty 0.03 -> 0.06; angular stays 0.25
Stage 16 combined bridge     angular/anchor 0.35/0.06; gates 1.42/0.127
Stage 17 combined commit     angular/anchor 0.35/0.06; gates 1.40/0.125
```

All four retain `conv_len>=0.92` and full rate `>=0.85`.  The angular positive
reward, target and sigma remain `0.20/1.25/0.35`; angular soft limit/scale
remain `1.30/0.50 rad/s`, and anchor sigma remains `0.10 m`.  Thus the
isolated trials identify reward strength without changing the requested
behavior or its coordinate.  Formal training enters Stage 16 only after both
isolated and combined candidates pass; it deliberately does not train through
the diagnostic-only Stage 14/15 sequence.  The accepted combined weights are
retained by every later stage.  V78's later stages remain in order, making
complete non-execution polish V79 Stage 26 and execution DR Stage 27 onward.
No stage lowers `conv_len` below `0.90`.

Bounded trials may use 512 environments/minibatch 8192 for 64 updates and
must use separate offline W&B identities.  Formal V79 uses 1024 environments,
minibatch 16384, rollout 128, two epochs, LR `5e-5`, gamma `0.9995`, lambda
`0.99`, clip `0.10`, target KL `0.004`, entropy `0.0002`, a 48-update window,
16-update warmup, stochastic 128-environment block validation and no stage
cap.  Frozen two-seed evaluation must confirm no regression in survival,
height, safety or failure-class distribution before formal launch.  The exact
preregistration is
`outputs/rl_sim/v79_quality_reward_repair_20260828/PREREGISTERED_TRIAL_PLAN.md`.

Canonical launchers are
`run_gpu0_measured_qvel_rmp_vertical_v79_resume_v78.sh` and
`run_gpu0_measured_qvel_rmp_vertical_v79_cgroup_resume_v78.sh`.  The GPU0-only
cgroup retains `MemoryHigh=11 GiB`, `MemoryMax=12 GiB` and
`MemorySwapMax=0`; GPU1 is outside the command and signal scope.  Formal W&B
continues `v72g0s7a1` with `resume=must`.  Its verified remote tail is
`3966500864`; offset `1369309184` places the first V79 rollout at
`3966631936` without modifying true checkpoint or optimizer time.

V79 bounded evidence rejected both sparse-event changes.  The angular-only
trial's last-24 rolling window kept `conv_len/full=0.92395/0.85690` but full
angular speed was `1.43725 rad/s`, not better than the V78 source rolling
`1.43585`; the anchor-only trial averaged `0.93066/0.85513` but its anchor
error worsened to `0.12704 m` and its last point fell to
`0.91425/0.83161`.  The combined V79 lesson was therefore never run.
Two-seed first-episode-per-lane frozen replay found V78 parent pooled
`full/angular/anchor=0.86328/1.45074/0.12510`; angular-trial last
`0.88672/1.45511/0.12557`; anchor-trial last
`0.88281/1.44678/0.12978`.  Neither candidate improved its named objective
without a quality trade.  V79 is a negative diagnostic profile and is not a
formal continuation.

The opt-in successor is
`goal_d455_measured_qvel_rmp_vertical_v80`.  It resumes the same exact V78
Stage-14 update-64 best and restores the original sparse angular/anchor event
weights `0.25/0.03`.  It changes only the already existing per-step
`full_norm` racket-stability angular penalty weight from
`0.42363636363636364 -> 0.50`; soft limit/scale remain
`0.5636363636363636/0.8 rad/s`.  This is a direct, causal pre-contact signal,
not another delayed event credit.

V80 has 32 stages.  Stage 14 is the dense-angular bridge with longevity
`0.92/0.85`, angular `<=1.45 rad/s` and anchor guardrail `<=0.130 m`; Stage 15
keeps the same domain/reward and tightens angular to `1.43`.  Later V78 stages
remain in order with the dense weight retained and no angular gate below
`1.43` or anchor gate below `0.130`.  Complete non-execution polish is Stage
24 and execution DR begins at Stage 25.  No DR, observation, reset, control,
height reward or PPO value changes, and no `conv_len` gate falls below 0.90.

The bounded V80 trial and rejection/fallback criteria are preregistered in
`outputs/rl_sim/v79_quality_reward_repair_20260828/PREREGISTERED_TRIAL_PLAN.md`.
Canonical launchers are
`run_gpu0_measured_qvel_rmp_vertical_v80_resume_v78.sh` and
`run_gpu0_measured_qvel_rmp_vertical_v80_cgroup_resume_v78.sh`.  Formal shape,
cgroup and W&B lineage remain the V79 values: 1024 environments, minibatch
16384, the 11/12 GiB zero-swap scope, W&B `v72g0s7a1`, `resume=must`, offset
`1369309184`.

The 64-update V80 bounded trial passed its rolling contract with last-24
hits/`conv_len`/full/angular/anchor
`13.521/0.92593/0.86190/1.42684/0.12419`; absolute apex/lift were
`1.29622/0.17046 m`, KL `0.00214`, qvel exceedance zero and qacc exceedance
`0.00426`.  On the preregistered two-seed, 256-episode frozen replay, V80
update 64 improved the V78 parent's full rate `0.86328 -> 0.89063`, mean
length `1114.18 -> 1131.32`, P10 length `705 -> 993`, early failures
`23 -> 15`, angular speed `1.45074 -> 1.44277 rad/s`, hit-racket VXY
`0.10091 -> 0.09821 m/s` and qacc exceedance
`0.004397 -> 0.004236`.  Hits remained in contract (`13.8281`), absolute
apex/lift remained `1.29768/0.17093 m`, and qvel exceedance remained zero.
Next-contact anchor error was `0.12890 m`: worse than the parent, but inside
the preregistered `0.130 m` guardrail.  It is not claimed as an improvement;
the rejected V79 isolation showed that raising its sparse reward made it
worse.

The frozen V80 evidence checkpoint is the bounded update-64 last, SHA-256
`8e8bce54a2e564cbbfef75bdcb743ff217dac48b77c65b926d53be225a0cf194`,
but it is not an eligible formal parent.  Formal V80 restores the pinned V78
1024-environment best and learns the accepted dense reward with a fresh
Stage-14 convergence window.  Detailed reward decomposition, failure-class
counts and the no-survival-regression decision are in
`outputs/rl_sim/v79_quality_reward_repair_20260828/V79_V80_REWARD_REPAIR_REPORT.md`.

Formal V80 started in tmux `pp_gpu0` at 2026-08-28 01:24:52 Asia/Shanghai
through the required cgroup launcher.  Its output directory is
`measured_qvel_rmp_vertical_v80_gpu0_seed20260943_20260828_dense_angular_repair_online1`.
JAX reported `CudaDevice(id=0)`, the exact V78 source restored at true step
`2597191680`, and training entered V80 Stage 14/32.  W&B reported `Resuming
run gpu0-v80-dense-angular-repair` for `v72g0s7a1`; the remote state became
`running` and its history step advanced beyond the expected first mapped
rollout.  At the recorded handoff the process had completed ten formal
1024-environment updates at about 19.6k steps/s with KL `0.00215`, explained
variance `0.9263`, qvel exceedance zero and qacc exceedance `0.00420`.  GPU1
remained outside the command and signal scope.

## GPU0 V81 Impossible Missing-Exposure Gate Repair

Formal V80 passed Stage 14 and Stage 15, then remained in Stage 16
`rmp80_nonexecution_inview_noise_support` for 4335 updates. This was not an
angular-reward failure. Stage 16 deliberately sets camera/view missing,
refresh dropout and burst dropout probabilities to zero, while it accidentally
inherits `min_ball_obs_missing_refresh_rate=0.006`. The measured refresh
missing rate was exactly zero in all 4335 rows, so the base convergence gate
was mathematically impossible and block validation was never called. Before
the resulting long overtraining tail, updates 169--229 passed every
policy-controlled gate for 61 consecutive updates. Update 200 rolling
hits/`conv_len`/full/angular/anchor were
`13.7029/0.93095/0.89164/1.41072/0.12596`; absolute apex/lift were
`1.29696/0.17217 m`, qvel exceedance was zero and qacc exceedance was
`0.00416`.

The opt-in successor is
`goal_d455_measured_qvel_rmp_vertical_v81`. It copies all 32 V80 stages and
changes only the minimum missing-refresh exposure gate: the gate becomes
`None` exactly when every mechanism capable of making a refresh missing is
disabled. Active missing stages retain their exact V80 lower bound. This
repairs Stages 16--32, not only the current stage. Reward, angular target and
weight, anchor reward and guardrail, survival gates, height contract, DR,
observation noise, control, reset, PPO and validation settings remain
V80-identical. In particular, the dense full-angular penalty stays `0.50`,
the Stage-16 angular/anchor gates stay `1.43 rad/s` and `0.130 m`, and the
Stage-16 `conv_len/full` gates stay `0.90/0.82`. The historical pass proves
these actor-controlled values are attainable without weakening survival, so
neither the angular reward nor its gate is changed.

V81 resumes only a frozen V80 Stage-16 checkpoint; it never replays Stages
1--15 and starts a fresh Stage-16 convergence window because the V80 history
contains the impossible gate. The leading source is update 200, true step
`2662072320`, Adam time `506920`, SHA-256
`3efc653bfd82d2af0d6da9a7d72bdce8672d111006b008d7166da10307559e83`.
Its exact selection and fixed-seed screen are registered in
`outputs/rl_sim/v81_missing_gate_repair_20260828/PREREGISTERED_V81_GATE_REPAIR.md`.
The canonical launchers are
`run_gpu0_measured_qvel_rmp_vertical_v81_resume_v80_stage16_update200.sh` and
`run_gpu0_measured_qvel_rmp_vertical_v81_cgroup_resume_v80_stage16_update200.sh`.
Formal shape remains 1024 environments, rollout 128, minibatch 16384, two
epochs, LR `5e-5`, clip `0.10`, target KL `0.004`, 128-lane block validation,
no stage cap and the 11/12 GiB zero-swap GPU0 scope. W&B continues
`v72g0s7a1` with `resume=must`; verified remote tail `4573364224` gives the
W&B-only offset `1911291904`, without changing the checkpoint's true step.
Formal launch must occur in tmux `pp_gpu0`; GPU1 remains outside command and
signal scope.

## GPU0 V82 Support-Aware `rec_next` Guardrail Repair

The current opt-in GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v82`. It resumes only the safely stopped
V81 checkpoint `mjx_curriculum_interrupted.pkl` at true step `2757230592`,
global update `726`, Stage-17 update `410`, Adam `t=516171`, and SHA-256
`951fa531fe05e1c64d9477f6977ba7799b349cdc2163280592a085f1d1dd0ef6`.
It must start a fresh
`rmp82_nonexecution_inview_ball_state_xy_bridge` convergence window and reject
curriculum-history resume, optimizer/critic reset, another source profile,
step, stage, stop reason, observation shape, or missing Adam/actor-anchor
state. Completed stages are not replayed.

Frozen 256-lane deterministic evidence showed that V81's unchanged
`0.130 m` next-contact-to-anchor gate became a signed target-coordinate
guardrail after the episode support widened: mean `rec_next` was `0.2391 m`
in the negative-X/negative-Y corner and `0.0795 m` in the
positive-X/positive-Y corner. Full-episode rates in those same cells remained
`0.923` and `0.842`; aggregate hits/length/full were
`13.97/1128.0/0.910`. Treat this as a narrow-domain gate copied into a wide
domain, not as evidence that the policy lost the ball. The environment metric
calculation is unchanged, and V82 does not change the 57-D actor observation
to expose a new target feature.

V82 copies all 32 V81 stages and preserves every reward, DR, target support,
control, observation, PPO, survival, height, angular, view and safety value.
It changes only the auxiliary `rec_next` guardrail and redundant minimum wait
windows from the current stage onward. The guardrail ladder is `0.165 m` at
the current +/-0.12 by +/-0.085 m bridge, `0.190 m` at the widest no-hold
lesson, `0.195 m` through non-execution target-hold/full polish, `0.200 m` for
RMP-internal lessons, `0.210 m` for PD/plant micro, and `0.220 m` for final
execution commit. Rolling-window and stochastic block validation remain
mandatory; no `conv_len` gate is below `0.90`. Do not reintroduce V79's
rejected sparse anchor-weight increase.

The preregistration and frozen episode evidence are under
`outputs/rl_sim/v82_rec_next_xy_bridge_repair_20260828/`. The formal launchers
are
`run_gpu0_measured_qvel_rmp_vertical_v82_resume_v81_stage17_update410.sh` and
`run_gpu0_measured_qvel_rmp_vertical_v82_cgroup_resume_v81_stage17_update410.sh`.
Formal training uses 1024 environments, minibatch 16384, an 11/12 GiB
zero-swap cgroup, and the explicitly reused W&B run `v72g0s7a1`. The remote
tail immediately before launch was `4668522496`; relative to the source true
step this gives the W&B-only offset `1911291904`. The bounded 512-env trial
kept hits/conv_len/full/rec_next at `13.5794/0.92210/0.87259/0.14879` through
64 updates, but did not advance because only 14/48 updates met the unchanged
64-completed-episode admission rule. Formal 1024-env training retains that
rule, the full 48-update window and block validation. V82 remains experimental,
simulation-only, and inherits all unresolved camera, ball-outcome and formal
RMP/PD evidence blockers.

## GPU0 V83 Stage-21 OOM-Boundary Continuation

The current GPU0 recovery continuation is
`goal_d455_measured_qvel_rmp_vertical_v83`, launched only through
`run_gpu0_measured_qvel_rmp_vertical_v83_cgroup_resume_v82_stage20.sh`.
V83 copies every one of V82's 32 stages exactly and changes no reward, domain,
gate, control, observation or PPO value. It resumes only V82's Stage-20 pass
checkpoint `20_rmp82_nonexecution_inview_target_hold_bridge_6p5.pkl` at true
step `2816999424`, Adam `t=521755`, SHA-256
`31d833cd31043664da183b93a26d7a89a4e104a187001f624f3a553a2b769e89`,
and enters fresh V83 Stage 21
`rmp83_nonexecution_inview_target_hold_bridge_6p5_polish`. It rejects another
source profile, step, actor/critic shape, optimizer state, domain, PPO shape,
curriculum-history resume, reset flags, or observation migration. Completed
Stages 1--20 are not replayed.

V82 Stage 20 passed at update 139 with rolling hits `13.6565`,
`conv_len=0.92`, full rate `0.87`, and `rec_next=0.16 m` against the
`0.195 m` gate. The subsequent Stage-21 reset compile took about 2 minutes 43
seconds and was killed before its first update. `systemd-oomd` recorded the
V82 scope as its top candidate with `11.3 GiB` use, pressure
`avg10=75.49%`, and a `11.3 GiB` peak; the old `MemoryHigh=11 GiB` induced
reclaim pressure even though the `12 GiB` hard limit was not reached. V83
retains 1024 environments and minibatch 16384, raises the GPU0 scope to
`MemoryHigh=13 GiB` and `MemoryMax=14 GiB`, retains zero cgroup swap, and sets
`ManagedOOMPreference=avoid`. This is an execution-scope repair, not a PPO or
curriculum change.

The reused W&B run is `v72g0s7a1` with `resume=must`. Its verified crashed
tail was `4728160256`, one rollout behind the Stage-20 boundary checkpoint.
V83 therefore uses the W&B-only offset `1911160832`, placing the first new
1024-by-128 rollout immediately after that tail without changing the true
checkpoint or optimizer step. The recovery audit is under
`outputs/rl_sim/v83_stage21_oom_resume_20260828/`.

## GPU0 V84 Stage-23/24 Survival-Ranking Repair

The opt-in successor is
`goal_d455_measured_qvel_rmp_vertical_v84`, launched only through
`run_gpu0_measured_qvel_rmp_vertical_v84_cgroup_resume_v83_stage23_update100.sh`.
It resumes only V83 Stage-23 update 100 at true step `2866544640`, Adam
`t=526571`, SHA-256
`c7ae82f871211825539412f952a305c3da5a23b9a87ac089fb0087498cf7401f`.
The 57-D actor, 368-D critic, Adam moments and Stage-23 domain are restored,
but V84 starts a fresh convergence window at
`rmp84_nonexecution_inview_wide_xy_stable_consolidation`. Curriculum-history
resume, another source profile/step/domain, CMDP, advantage reweighting,
optimizer/critic reset and observation migration fail closed.

V83 Stage 23 and Stage 24 have identical environment/reward configurations;
only their gates differ. During V83 Stage 23, raw early-to-late mean length,
full rate and hits fell by about `13.5 steps / 0.0130 / 0.175`, while return
rose about `0.98`. Frozen deterministic first-episode replay of 256 identical
lanes measured update-100 versus interrupted update-343 hits/full/length
`14.145/0.9102/1136.4` versus `14.066/0.9023/1132.5`. Ten lanes changed from
full to failure and eight from failure to full; lower-Y view termination was
the dominant incremental cause. Exact additive attribution showed the later
policy gained `+2.035` reward per episode by reducing the low-apex penalty,
more than its `+0.500` total-return advantage, while longevity worsened.

From Stage 23 onward V84 changes only:

```text
low_hit_penalty_weight             7000 -> 5000 m^-2
full_episode_completion_reward        0 -> 40
full_episode_completion_min_hits       0 -> 13
```

The 1.28 m absolute-apex lower edge, existing weight-20 joint apex/lift
completion event, dense survival, all other rewards, every gate, PPO value,
control, observation, reset, physics and DR range remain unchanged. Stage 23
is the repair bridge, Stage 24 is the unchanged-domain full-episode polish,
and the committed reward persists into Stage 25 and later RMP/PD DR lessons.
The Stage-23/24 `conv_len/full` gates remain `0.92/0.84` and `0.95/0.90`;
being numerically above the lower Stage-23 full gate does not count as
acceptance when the metric is declining.

Before formal promotion, the preregistered bounded run must use 512
environments, 128 rollout steps, minibatch 8192, the unchanged conservative
V83 PPO scalars, at most 64 Stage-23 updates and offline W&B. Both `conv_len`
and full must be non-declining and no worse than the source while height,
cadence, view, motion quality and qvel/qacc safety remain in range. Formal
training, if accepted, uses 1024 environments, minibatch 16384, no stage
update cap, W&B `v72g0s7a1` with `resume=must`, offset `1943011328`, and the
default 13/14 GiB zero-swap cgroup. V84 remains experimental, simulation-only
and inherits every unresolved real-camera, replacement-ball outcome and
formal RMP/PD coverage blocker.

The bounded 64-update V84 run was directionally inconclusive and is not the
formal continuation.  Its final rolling `full=0.8528` remained above the
Stage-23 gate and its last 16-update slope was positive, but frozen 128-lane
same-seed replay measured source/V84 `full=0.8516/0.8594` while mean length
fell `1101.5 -> 1088.2` steps and failed-episode median length fell
`428 -> 244` steps.  A sparse step-1200 completion event can slightly raise
the probability of truncation without distinguishing an early failure from a
late one.  Do not present its short late-window rise as sustained evidence.

## GPU0 V85 Stage-23/24 Failure-Time Survival Repair

The opt-in successor is
`goal_d455_measured_qvel_rmp_vertical_v85`, launched only through
`run_gpu0_measured_qvel_rmp_vertical_v85_cgroup_resume_v83_stage23_update100.sh`.
It resumes the same exact V83 Stage-23 update-100 checkpoint and preserves
V84 stages, gates, domains, control, observations, physics, DR, PPO and every
other reward.  From Stage 23 onward it adds only:

```text
early_termination_remaining_horizon_penalty = 30
terminal event = -30 * (episode_limit - failure_step) / episode_limit
```

The event is zero on a normal 1200-step truncation.  Its 30-point bound is the
exact maximum full-horizon scale of the existing post-first-hit alive credit:
`5 reward/s * 0.005 s/step * 1200 steps = 30`.  V84's 40-point, >=13-hit
completion event remains responsible for `full`; the new event distinguishes
the failure-time distribution that determines `conv_len`.  Because the event
is emitted at the actual failure, finite-lambda GAE can assign it to the
preceding causal contact cycle instead of attempting to propagate a step-1200
event hundreds of steps backward.  This is a bounded objective-alignment
change, not CMDP, failure oversampling or an update-count workaround.

Promotion requires one bounded 512-env, at-most-64-update Stage-23 trial from
the pinned source.  The registered direction passes only if the 48-update
rolling survival window does not exhibit the V83/V84 joint decline, the late
window is not declining, and frozen paired replay shows no earlier-failure
shift while height, cadence, view, motion quality, qvel/qacc and PPO KL remain
valid.  If accepted, formal training uses 1024 environments, minibatch 16384,
W&B `v72g0s7a1` with `resume=must`, offset `1943011328`, no stage update cap,
and the 13/14 GiB zero-swap cgroup.  V85 remains experimental,
simulation-only and inherits every unresolved deployment evidence blocker.

The preregistered 64-update V85 bounded trial was accepted for direction.
All three post-warmup 16-update windows had positive within-window slopes for
both raw mean length and full rate; the full updates-17--64 slopes were
`+0.3296 steps/update` and `+0.000659/update`.  Final 48-update rolling metrics
were `conv_len=0.921758`, `full=0.866534`, hits `13.8121`, absolute apex
`1.292412 m`, lift `0.164653 m`, mean/RMS racket VXY
`0.107231/0.119998 m/s`, full angular speed `1.323304 rad/s`, qvel exceedance
zero, qacc exceedance `0.004293`, and mean exact KL `0.002119`.  This supports
the reward direction under the requested short experiment; it is not evidence
that future formal curves cannot oscillate.  Formal training must restart
from the pinned V83 source, not from the 512-env trial checkpoint.

## GPU0 V86 Stage-24 Monotone-Survival Transaction Repair

The opt-in successor is
`goal_d455_measured_qvel_rmp_vertical_v86`, launched only through
`run_gpu0_measured_qvel_rmp_vertical_v86_cgroup_resume_v85_stage23_pass.sh`.
It resumes V85's Stage-23 pass-boundary checkpoint at true step `2900099072`,
Adam time `529643`, SHA-256
`e48c5b50ebfea05df3cd44748eaa13094ce3f0409315a32b31d4914185c0383d`,
and starts a fresh Stage-24 window. V85's Stage-24 best/last checkpoints are
negative continuation evidence and are prohibited sources.

The complete 539-row V85 audit found that Stage 23's rolling
`conv_len/full` peaked at `0.926784/0.875329` and ended at
`0.922111/0.859691`. Stage 24 never approached its `0.95/0.90` gate and
ended at `0.902329/0.832832`; after its first full 48-update window, raw
length/full slopes were `-6.37858e-5/-3.63998e-5` per update. Stage-24
termination probability rose by `0.00772`, led by ball-view Y-low and X-high
failures, while hit-racket mean/RMS VXY rose by about `0.0043/0.0044 m/s`.
Exact KL stayed near `0.0027` below the `0.004` target, explained variance
rose, qvel exceedance was zero and qacc exceedance stayed near `0.0049`.
This is continued policy/optimizer drift on an unchanged domain, not PPO
explosion, a safety stop or a new DR boundary.

V86 preserves all 32 V85 stages and changes no environment, reward, gate,
control, observation, physics, DR or PPO field. Its only optimization-policy
change is mandatory atomic transaction promotion:

```text
candidate block                         = 16 PPO updates
paired evaluation                       = 256 deterministic first episodes
paired seed                             = 20261004
accepted while below Stage-24 gates     = combined survival gap strictly falls
individual conv_len/full gap increase   = prohibited
individual conv_len/full value decrease = prohibited, including above gates
secondary/safety gate-gap increase      = prohibited
qvel/qacc exceedance ceilings           = 0.0 / 0.01
commit unit                             = actor + critic + Adam moments
```

Rejected blocks restore the incumbent state and do not enter convergence
history, best-checkpoint ranking or periodic checkpoint state. Once both
primary metrics are in range, accepted incumbents must remain in range. This
makes survival lexicographically prior to return-shaping trades without
retuning the V85 reward or weakening Stage-24's `0.95/0.90` contract.

Promotion requires a separate offline 512-environment, minibatch-8192,
32-update trial from the pinned source. At least one transaction must be
accepted, every accepted incumbent must be componentwise non-regressing on
the paired survival population, and all secondary/safety gates must remain
non-worse. The formal run restarts the same source with 1024 environments,
minibatch 16384, LR `5e-5`, two epochs, clip `0.10`, target KL `0.004`, a
48-update convergence window, no stage cap, a new W&B ID/history and the
13/14 GiB zero-swap cgroup. V86 remains experimental and simulation-only.

The 2026-08-28 bounded GPU0 evidence satisfied this promotion contract. With
the strict componentwise rule, trial 2 rejected candidates
`0.92969/0.87891 -> 0.92380/0.85938` and
`0.92969/0.87891 -> 0.93615/0.87500`. Independent trial 3 rejected
`0.94306/0.89844 -> 0.92963/0.87891` at update 16 and accepted
`0.94306/0.89844 -> 0.94323/0.89844` at update 32. The accepted candidate
had zero qvel exceedance, qacc exceedance `0.004870`, and no secondary gate
regression. The formal run must restart the pinned V85 checkpoint; neither
bounded trial is a formal resume source.

The formal V86 run started on 2026-08-28 in `pp_gpu0:v86_train` with output
identity
`measured_qvel_rmp_vertical_v86_gpu0_seed20261005_20260828_stage24_monotone_transaction_online1`
and W&B ID `v86g0txn1`. Startup confirmed JAX `CudaDevice(id=0)`, the pinned
source at true step `2900099072`, Stage 24, actor/critic dimensions `57/368`
and 1024 environments. Its initial paired incumbent is
`conv_len/full=0.93621/0.88281`.

## GPU1 V129 Result and V130 Residual-Recovery Successor

The original
`formal_gpu1_v129_geometric_survival_20260825` run stopped safely at Stage 23,
update 537, true step `9869033472` when the host-memory guard reported
`swap_growth_under_pressure`. Its interrupted/last checkpoint SHA-256 is
`ccffab6fc6cbac204a7362628a4b58dd13a48939c1359688158c9420d9abc22e`; the
causal progress CSV SHA-256 is
`65e87b96e7ba1be62d45085a1d36c7e5ad3e737ad4c5774d13a83833fbc3f6a3`.
The payload contains the 67-D actor, 279-D critic, Adam time `930201`, exact
stage-local actor anchor and Stage-23 convergence history.

GPU0's V67 range decision is not copied blindly. V129 already has exact-zero
ball-spin support on every axis, while the frozen GPU0 half-solref screen was
weaker than half spin; therefore GPU1 reward, contact/solref/actuator DR,
reset distribution, observations and gates remain unchanged. The transferable
lesson is the stable 131072-sample concurrent resource envelope. The reviewed
recovery launcher
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v129_resume_interrupted_nenv512.sh`
keeps rollout 256 and all PPO scalars but changes environments `1024 -> 512`,
halving batch `262144 -> 131072`. Minibatch 16384 and two epochs are retained,
so optimizer work per environment sample remains unchanged. It restores
critic, Adam, true/global/stage steps and all 537 causal history rows; reset
flags and observation migration are prohibited by the launcher contract.

The recovery output was
`outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/formal_gpu1_v129_geometric_survival_resume1_nenv512_20260826`
with seed `82731`, physical GPU1 UUID
`GPU-e74458ec-002a-1bac-e0be-d0a5713b661e` and new offline W&B ID
`v129g1r1`. It eventually stopped safely at update 2521 because of sustained
low host-available memory. The interrupted checkpoint is
`mjx_curriculum_interrupted.pkl`, true step `10129080320`, SHA-256
`7133a12f2076ba680adae2b126cacf463fe67199e8c400e4fa7712fc345ceb40`.
Its final complete 48-update window was hits `11.8535`, episode-length fraction
`0.9052`, full-episode rate `0.8105`, true-view fraction `0.9437`, relative apex
`0.2212 m`, exact KL `0.00164`, qvel exceedance `0.000058` and qacc exceedance
`0.00293`. This is long, stable negative evidence: geometric-survival weight
3.0 did not improve the primary `conv_len/full` plateau and V129 must not be
resumed or promoted.

V130 `goal_d455_sport_taskspace_record_new3_sim2real_residual_recovery_v130`
uses the GPU0 transferable result that successful continuations first
reconsolidated a strong policy in a smaller domain and only then introduced
the hardest execution uncertainty. It resumes the immutable V123 update-8400
checkpoint (SHA-256
`592f1a1586e08aa0181f10284c42b977e5876bede078c33bcb3d107a0b027319`)
instead of the degraded V129 tail and also uses that selected source as the
actor anchor. V123 reward, PPO, observation/action, contact, spin, reset and
final residual-100 support remain exact. Starting at zero-based Stage 22, V130
uses four uncapped, block-validated lessons: residual 75 recovery; residual
87.5 bridge; full continuous frequency/damping/gain residual ranges while
retaining delay `[-3,+2]`; then the exact residual-100 delay `[-3,+3]`
commit. Their `conv_len` gate is `0.92`, while full-episode gates are
`0.82/0.82/0.82/0.83`; the unchanged V123 hold tail follows in order.

The reviewed formal launcher is
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v130_residual_recovery.sh`.
It fixes physical GPU1, seed `82731`, 1024 environments, rollout 256, batch
262144, minibatch 16384, two epochs, W&B online with a fresh run ID, a new
output directory, no stage cap and no optimizer/critic reset. V130 remains
simulation-only; graduation requires every rolling and block-validation gate,
not merely launch health or higher return.

The 1024-environment run passed residual-75, residual-87.5 and the continuous
residual-100 bridge, then stopped safely in residual-100 commit Stage 26 at
stage update 38/global update 644 because the host-memory guard detected swap
growth under pressure. Its interrupted checkpoint has true step `9897082880`,
SHA-256 `8f8d6001bbe1003196694d1484887f0e0b9c70dc6fe040c66cd64a4f929ec7ab`;
the 644-row causal progress CSV SHA-256 is
`f042c17fe7c01670ce054107af68dbb8f30aeb13de113cd322d1f85223e5896c`.
The user-authorized resource continuation is
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v130_resume_interrupted_nsteps128.sh`.
It restores actor, critic, Adam, stage-local anchor, true/global/stage steps and
the complete convergence history, keeps 1024 environments, and changes only
rollout length `256 -> 128`, reducing batch `262144 -> 131072`. Rewards, DR,
gates and every other PPO scalar remain exact. It uses a new output directory
and W&B-online identity; optimizer/critic reset and observation migration are
prohibited. This resource choice may later be reversed by another explicit
continuation if the shorter GAE rollout degrades task evidence.

The n_steps-128 continuation later stopped safely on SIGINT at true step
`10062626816`, global update `1907`, Stage 31
`record_new3_sim2real_v130_hold_8pct_tail_half` update 91. The saved row had
already passed the stage (`hold=12/12`, `stage_converged=1`); checkpoint
SHA-256 is `5ced698f5fba965c5bf030ac8b45921c46c065ef078f179a0435e0951c994d88`
and the complete 1263-row progress CSV SHA-256 is
`ce7094f9220c7300e5efb2cb15dfddea8f93fad4a1ad69ff74fc52bb00bf9d7c`.
An initial `run_gpu1_record_new3_v130_resume2_nsteps128.sh` attempt incorrectly
restored that already-converged Stage-31 history and created W&B ID `v130g1r2`;
it was stopped and its output/checkpoints are rejected continuation sources.
The corrected launcher uses the same pinned pre-attempt checkpoint but starts
directly at Stage 32,
`record_new3_sim2real_v130_hold_8pct_tail_half_consolidate`, with
`stage_update=0` and a fresh convergence window. It restores actor, critic,
Adam and the true checkpoint environment step, keeps the 1024-by-128 batch and
all V130 reward, gate, DR and PPO scalars unchanged, and appends to the
existing W&B-online run `v130g1r1` with `resume=must` and zero W&B step offset.
Optimizer/critic reset, observation migration, reuse of the rejected attempt,
and replay of Stage 31 are prohibited.

## GPU1 V131 Spinless Observation-Rate Polish Continuation

V130 later completed one-based Stage 34,
`record_new3_sim2real_v130_hold_8pct_tail_full_consolidate`, then regressed
immediately when Stage 35 introduced latent reset X-spin together with impact
DR. The stopped Stage-35 continuation reached update 682 and true step
`10248093696`; its rolling hits/episode-length/full rates were approximately
`9.52/0.75/0.65`, compared with the completed Stage-34 tail near
`12.07/0.92/0.84`. It stopped safely on SIGINT and is negative evidence; do
not resume its actor, critic or optimizer.

The opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_spinless_polish_v131`,
launched only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v131_spinless_stage35_from_v130_stage34.sh`.
It resumes the Stage-34 boundary checkpoint
`34_record_new3_sim2real_v130_hold_8pct_tail_full_consolidate.pkl` at true
step `10120822784`, Adam time `960897`, SHA-256
`d3c87a12fc14765eb26706ddb4c8af2703a8b2206e5277404904d40042929988`.
This checkpoint retains the 67-D actor, 279-D critic and all Adam moments.
V131 starts one-based Stage 35 with fresh stage/convergence history; it must
not restore V130 Stage-35 history, reset optimizer/critic, migrate observation
dimensions, or resume another source.

V131 preserves V130 Stages 1--34 exactly. From Stage 35 onward it changes only
explicit reset-spin/normalized-inertia DR and observation cadence: all three
reset-spin intervals are exactly zero, normalized inertia is fixed at `0.40`,
and fractional scheduling is enabled so requested 60/90 Hz rates are exact on
average. Natural contact-generated angular velocity remains in MuJoCo; this
profile does not force the physical ball to remain non-rotating. Impact
friction, solref, racket geometry, latency, actuator DR, reward, gates, resets,
observations, action semantics and PPO scalars otherwise remain V130-exact.

The post-boundary schedule is:

| One-based stages | Ball-state cadence | Purpose |
| --- | --- | --- |
| 35--40 | exact 60 Hz | no-spin impact-half/full and latency adaptation plus consolidation |
| 41--42 | exact 90 Hz | direct 90 Hz adaptation and consolidation; no 75 Hz bridge |
| 43--44 | exact 60 Hz | return-to-deployment-rate polish and consolidation |
| 45--47 | exact 60 Hz | low/high-energy recovery and final combined proof |

The formal launcher fixes physical GPU1, seed `82731`, 1024 environments,
rollout 128, minibatch 16384, two epochs, LR `3e-5`, clip `0.08`, target KL
`0.003`, no stage update cap and block validation. By explicit user direction,
it appends to W&B-online run `v130g1r1` with `resume=must` despite the new
profile identity. The W&B-only offset is `128974848`: this places the first
V131 update one 131072-sample rollout after the stopped V130 W&B tail while
leaving the checkpoint's true optimizer/global step unchanged. Output goes to
a new V131 directory and never overwrites a V130 artifact.

Acceptance remains the complete existing gate set. At minimum, Stage 35 must
recover rolling hits `>=11.5`, `conv_len>=0.90`, full rate `>=0.65`, view,
apex, cadence, contact-quality and safety gates for its 12-update hold before
advancing. The final proof still requires the V130 Stage-47 hit, survival,
full, view and safety contract; easier 90 Hz performance is not final evidence
because Stages 43--47 return to 60 Hz.

## GPU1 V132 Post-Hit Ball-Latency Bridge Continuation

V131 passed one-based Stage 38,
`record_new3_sim2real_v131_impact_full_no_spin_60hz_consolidate`, but its
direct Stage-39 latency transition collapsed from the passed Stage-38
distribution (about 12 hits, `conv_len=0.92`, full `0.85`) to a long plateau
near 8 hits, `conv_len=0.63` and full `0.40`. PPO KL and explained variance
remained healthy. The executable delta showed the cause was a task jump:
`dr_randomize_latency` sampled 0--2 steps once per reset and delayed the whole
base observation, including joint/base/racket/action-derived state, rather
than modeling intermittent ball-camera latency during juggling.

The opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_post_hit_latency_v132`, with
launcher
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v132_post_hit_latency_from_v131_stage38.sh`.
It resumes only the completed V131 Stage-38 boundary archive at true step
`10828742656`, Adam time `1047298`, SHA-256
`e436e690b28d0b4bdb8d907e1cdcf459bb769a9814f18dd982ead7577d7afedb`.
It restores the 67-D actor, 279-D critic and Adam moments, starts V132 Stage 39
with a fresh convergence window, and prohibits curriculum-history,
optimizer/critic reset, observation migration or the negative V131 Stage-39
checkpoint.

V132 preserves V131 Stages 1--38 exactly and replaces V131 Stages 39--40 with
five 60 Hz lessons. At every fresh ball observation after the first confirmed
hit, the lesson independently activates with probability `0.25 -> 0.50 ->
0.75 -> 1.00 -> 1.00`; an active frame samples the existing 0/1/2-control-tick
support, i.e. 0/5/10 ms. Before the first hit the injected latency is always
zero. Only ball position, ball velocity, ball/racket relative position and
ball age are buffered; q/dq, base state, racket state and action feedback stay
current. Because the sampled support contains zero, the actual nonzero-delay
fractions are `1/6`, `1/3`, `1/2`, `2/3`, `2/3`, respectively.

The bridge and remaining schedule are:

| V132 stages | Cadence | Fresh post-hit activation | Survival weight | Purpose |
| --- | --- | --- | --- | --- |
| 39--42 | exact 60 Hz | 25%, 50%, 75%, 100% | 2.10, 2.20, 2.30, 2.40 | gradual latency adaptation |
| 43 | exact 60 Hz | 100% | 2.40 | latency consolidation |
| 44--45 | exact 90 Hz | 100% | 2.40 | direct 90 Hz adaptation/consolidation |
| 46--47 | exact 60 Hz | 100% | 2.40 | deployment-rate polish/consolidation |
| 48--50 | exact 60 Hz | 100% | 2.40 | energy recovery and final proof |

The reward repair is deliberately bounded: the existing dense geometric
`post_hit_survival_reward_weight` ramps from V131's `2.0` to `2.4`, while the
next-contact anchor penalty stays at the passed Stage-38 value `0.60` instead
of taking V131 Stage 39's simultaneous reduction to `0.50`. It does not add an
unconditional alive bonus, change hit-event credit, or relax any gate. Stages
39--42 keep the V131 learning gate (`hits>=11.5`,
`conv_len>=0.90`, full `>=0.65`, view `>=0.90`); Stage 43 keeps the V131
consolidation gate (`hits>=12.0`, `conv_len>=0.92`, full `>=0.70`, view
`>=0.94`). Downstream gates, no-spin/inertia contract, contact/actuator DR,
control, reset distribution and PPO settings remain unchanged.

The formal launcher keeps seed `82731`, 1024 environments, rollout 128,
minibatch 16384, two epochs, LR `3e-5`, clip `0.08`, target KL `0.003`, no
stage-update cap, block validation and W&B online. By explicit user direction,
V132 appends to `v130g1r1` with `resume=must` despite the changed reward and
observation-DR semantics. Its W&B-only offset is `355074048`, placing the first
V132 rollout one 131072-sample point after V131's final history point while
leaving the Stage-38 checkpoint and optimizer step unchanged. V132 still uses
a new output directory and never overwrites V131 artifacts.

Stage 41 (`latency_p75_60hz_bridge`) subsequently passed. The host-memory
guard stopped safely during the first Stage-42 update after accumulated JAX
stage compilations increased swap use. The user-authorized recovery uses the
same launcher with `V132_RESUME_FROM_STAGE41=YES`, restores the Stage-41
boundary at true step `10876059648`, Adam time `1053074`, SHA-256
`7778275b59e5e46dc177792015f75aec258237f00fa80ceafd87208b7ab937cd`,
and starts Stage 42 (`latency_p100_60hz_commit`) at update zero with fresh
convergence history. It deliberately discards the one interrupted Stage-42
update, clears JAX caches between later stages, continues W&B `v130g1r1` with
W&B-only offset `355205120`, and writes a new output directory.

## GPU1 V133 Late-Survival Reward Repair

V132 Stage 42,
`record_new3_sim2real_v132_post_hit_ball_latency_p100_60hz_commit`,
remained at the same plateau for 1944 updates: its terminal 48-update window
was approximately 11.35 hits, `conv_len=0.88` and full rate `0.76`.
PPO remained stable (low KL, high explained variance, no raw-action clipping
or qvel failure), while the dominant failed-episode exits were racket high,
ball low and horizontal view-bound loss. The profile's bounded hit floor
contributed most positive reward. Its geometric post-hit survival term turned
off during late descent, its direct after-hit alive and completion terms were
zero, every termination per-hit coefficient was zero, and the racket-z soft
barrier was disabled. Increasing only fixed termination cost (V125) or only
direct alive weight from 3 to 5 (V126/V127) is already preserved negative
evidence and is not repeated.

The opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_survival_repair_v133`,
launched only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v133_survival_repair_from_v132_stage41.sh`.
It resumes the clean V132 Stage-41 boundary at true step `10876059648`,
Adam time `1053074`, SHA-256
`7778275b59e5e46dc177792015f75aec258237f00fa80ceafd87208b7ab937cd`.
It restores the 67-D actor, 279-D critic and all Adam moments, starts a fresh
Stage-42/convergence window, and deliberately rejects the plateaued V132
Stage-42 actor.

V133 preserves V132 Stages 1--41 exactly. From Stage 42 onward it changes only
these reward fields:

```text
post_first_hit_alive_reward_weight:          0 -> 3
full_episode_completion_reward:              0 -> 20
full_episode_completion_min_hits:            0 -> 10
termination_miss_penalty_per_hit:            0 -> 0.75
racket_z_limit_termination_penalty_per_hit:  0 -> 0.75
racket_anchor_termination_penalty_per_hit:   0 -> 0.75
racket_z_band_up/down:                       +0.20/-0.00 -> +0.12/-0.04 m
racket_z_soft_penalty_weight:                0 -> 60
```

The direct alive term is still horizon-bounded (at most about 18 reward over
six seconds), and the 20 completion event requires ten counted hits, so a
never-acquired or stationary episode cannot collect it. Per-hit terminal
cost makes an otherwise identical late failure worse than an early failure.
The local racket-z term is the trace-supported near-boundary form used by the
independent GPU0 V78 experiment; V133 does not copy GPU0 control or dynamics,
does not change the hard racket limits, and leaves the global upward-drift
penalty at zero. Latency, cadence, observations, action semantics, actuator
physics/DR, contact, reset distribution, all graduation gates and every PPO
setting remain V132-exact.

The exact 512-environment, 64-update bounded trial is
`v133_survival_reward_repair_20260827/stage42_trial64_gpu1_seed82731`
with offline W&B ID `v133g1s42trial1`. It failed the preregistered final
window: hits `11.517` passed, but `conv_len=0.8877` and full
`0.7751` missed `0.90/0.80`. Mean exact KL was healthy at `0.00202`,
qacc exceedance was `0.00271`, view was about `0.92`, and apex stayed
about `0.222 m`. Alive/completion reward contributed about
`0.01368/0.01317` per step, proving the new terms were active; the racket-z
soft term remained local at only about `-1.57e-5` per step. Racket-high
frequency improved only slightly while ball-low frequency worsened. V133 is
negative evidence: do not formal-launch it or resume its trial checkpoint.

## GPU1 V134 Rejected Direction

V134's proposed latency-probability microbridges were rejected by the user on
2026-08-27 before the trial produced an optimizer update. No executable V134
profile or launcher is retained, and its startup-only output is not a
checkpoint source. The next successor must address the observation-latency
semantics directly rather than add curriculum stages.

## GPU1 V135 Timestamp-Consistent Ball-Latency Continuation

The opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_timestamp_latency_v135`,
launched only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v135_timestamp_latency_from_v132_stage41.sh`.
It resumes the same clean, naturally passed V132 Stage-41 boundary at true
step `10876059648`, Adam time `1053074`, SHA-256
`7778275b59e5e46dc177792015f75aec258237f00fa80ceafd87208b7ab937cd`.
It restores the 67-D actor, 279-D critic and all Adam moments, starts V135
Stage 42 with a fresh 16-update warm-up and convergence window, and rejects
curriculum-history resume, optimizer/critic reset, observation migration and
every bounded-trial checkpoint. V134 is not reintroduced.

V132/V133 selected delayed ball values from a buffer advanced at 200 Hz, but
labeled them with only the sampled 0/5/10 ms delay. At exact 60 Hz, a selected
buffer entry can belong to the preceding camera frame and continues aging
between refreshes. The actor therefore received a delayed position/velocity
whose `ball_obs_age` contradicted its acquisition time. Since the 67-D actor
has no separate injected-latency feature, that mismatch made the post-hit
state partially unobservable.

V135 preserves all 50 stages and keeps Stages 1--41 exactly equal to V133.
It enters p100 directly at Stage 42; no bridge or micro-bridge stage is added.
From Stage 42 onward, the delayed ball position/velocity, ball-racket relative
position and actor age share the selected camera sample's timestamp. A fresh
sample has age zero, a held 60 Hz frame ages at each 5 ms control tick, and a
selected 0/1/2-tick older frame adds its actual history age. Pre-hit injected
latency remains zero. The feature dimension, normalization/clipping, p100
probability, 0/1/2-tick support, reward, gates, control, physics, DR, resets
and PPO values are otherwise unchanged from V133.

The first clean 512-environment, 64-update bounded trial is
`v135_timestamp_latency_repair_20260827/stage42_trial64_gpu1_seed82731_run2`
with offline W&B ID `v135g1s42trial2`. It proved that direct p100 activation
is trainable: after the 16-update warm-up it recovered to about 12 hits and at
its best reached rolling `conv_len/full` near `0.92/0.82`. Its terminal window
was hits `11.6895`, `conv_len=0.89307`, full `0.78503`, view `0.92540` and
relative apex `0.22220 m`; mean actor ball age was about `10.05 ms`, exact KL
about `0.00115`, qacc exceedance about `0.00288`, and raw-action clipping zero.
It therefore passed task/PPO/safety health but missed the preregistered final
`conv_len/full=0.90/0.80` promotion pair. Do not use that checkpoint as the
formal continuation source. A clean 84-update repeat with the same large
minibatch plateaued at `11.54261/0.88194/0.76989`. A matched formal
optimizer-step trial using minibatch 8192 retained coefficient `0.01` and
ended at `11.68630/0.88999/0.78033`; its last-12 actor-anchor regularization
was `0.00469`, several times the PPO policy-loss scale, while anchor KL was
being driven toward an old V123 residual-100 actor. This identified excessive
anchor pressure as a continuation-specific conflict rather than a need for
another curriculum bridge.

The final one-variable confirmation
`stage42_trial64_mb8192_anchor002_gpu1_seed82731`, offline W&B
`v135g1s42a002`, changes only the anchor coefficient from `0.01` to `0.002`.
Its terminal rolling hits/`conv_len`/full were
`11.78915/0.90033/0.79765`, view `0.93363`, relative apex `0.22322 m`; last-12
exact KL, clip fraction, qacc and qvel exceedance were respectively
`0.00200/0.18474/0.00262/0.000062`. The executable strict stage gate passed.
The auxiliary full-rate target remained only `0.00235` below exact `0.80`.
The user directed that bounded tuning stop here, so formal training uses the
better `0.002` setting with no stage-update cap. A mid-stage environment
restart must still reset/exclude its warm-up rather than append cold-reset
episodes to an existing window, and no bounded checkpoint is a formal source.

Formal training keeps 1024 environments, 128 rollout steps, minibatch 16384,
two epochs, LR `3e-5`, clip `0.08`, target KL `0.003`, actor-anchor coefficient
`0.002`, no stage-update cap and block validation. By explicit user direction
it appends to W&B-online run `v130g1r1` with `resume=must`. The
W&B-only offset `610009088` places the first clean V135 rollout at
`11486199808`, one 131072-sample point after the last local V132 history point,
without changing true checkpoint or Adam time. Output must use a new V135
directory and formal launch must occur inside tmux session `pp_gpu1`.

V135 remains simulation-only. The current real simcode5/simcode6 paths force
valid-ball-age use off and are not compatible merely because the actor stays
67-D. Before hardware use, implement and validate the separate timestamp,
prediction, loss and reacquisition contract in `AGENTS.md`; short post-hit
latency DR is not evidence for long out-of-view prediction or arbitrary
dropout.

## GPU1 V136 Stage-43 Gate-Calibrated Continuation

The current opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_gate_calibrated_v136`,
launched only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v136_gate_calibrated_from_v135_stage43.sh`.
V135 Stage 43 was safely stopped after 6554 updates at true step
`11839438848`; its rolling hits/length/full/view had deteriorated to
`11.51411/0.90019/0.79597/0.93297`. Across the entire Stage-43 history, rolling
hits peaked at `12.17780`, length peaked at `0.91291`, and the configured 0.92
length target passed zero times. Because Stage 42 and Stage 43 have identical
configuration and differ only in gates, this plateau is not evidence of a new
DR discontinuity or insufficient network capacity.

The selected parent is the V135 Stage-43 update-5400 archive, true step
`11688181760`, Adam time `1152135`, SHA-256
`67c731bb518b051910508bfd3d3a8a37743eaadc309edd46cb3fd6105b23bed7`.
Its historical rolling hits/length/full/view/apex were
`12.01235/0.91140/0.81479/0.93949/0.22285 m`. Frozen deterministic
128-episode validation measured `12.15625/0.91665/0.83594/0.94789`, with qacc
exceedance `0.00389`; the stopped checkpoint was worse on hits, apex and view.
V136 restores actor, critic and Adam moments from update 5400, starts fresh
Stage-43/convergence history with a 16-update warm-up, and rejects curriculum
history resume, optimizer/critic reset, observation migration, the stopped
checkpoint and both bounded-trial actors.

V136 has the same 50 stages as V135 and preserves Stages 1--42 exactly. Stage
43 alone changes rolling hits from `12.0` to `11.8` and rolling length fraction
from `0.92` to `0.91`. Full-episode rate `0.70`, view `0.94`, apex
`0.218--0.248 m`, hit1/hit3/hit12, hits-ge3, cadence, recoverability, motion
and safety gates remain unchanged. It remains stricter than Stage 42's
`11.5/0.90/0.65/0.90` contract. Recomputing all recorded leaf gates showed a
17-update completely passing interval at V135 updates 2515--2531 for
`11.8/0.91`, meeting the unchanged 16-update hold. At `12.0/0.91` the longest
joint run was only two updates; at `12.0/0.92` there were none. Every Stage
44--50 gate remains exactly as in V135, including subsequent `12.0/0.92`
consolidations and the final proof.

The evidence directory is
`outputs/rl_sim/v136_stage43_survival_gate_repair_20260828/`. Its first
512-environment diagnostic reduced actor-anchor coefficient `0.002 -> 0.0005`;
the terminal window was hits `11.7688`, length about `0.91`, full `0.81`, view
about `0.94`, apex `0.217 m`, and actor-anchor KL `1.61`, so the change was
rejected. A second bounded diagnostic conditioned a 30-point completion reward
on at least 12 hits. It briefly reached a fully passing
`12.2615/0.93/0.85/0.94` joint window, but ended at
`11.7465/0.90/0.80/0.94` with anchor KL `1.85`; that reward change is also not
promoted. These negative trials show that continued parameter pressure causes
policy drift rather than a stable Stage-43 improvement.

Formal V136 therefore preserves the V135 reward, DR and PPO settings: 1024
environments, 128 rollout steps, minibatch 16384, two epochs, LR `3e-5`, clip
`0.08`, target KL `0.003`, anchor coefficient `0.002`, no stage-update cap and
block validation. It runs in tmux `pp_gpu1` and appends to W&B-online run
`v130g1r1` with `resume=must`. W&B-only offset `761266176` places the first
V136 point one 131072-step rollout after the final V135 W&B point while the
true checkpoint and Adam time remain unchanged. V136 is simulation-only and
inherits the V135 ball-age/prediction deployment blockers.

## GPU1 V137 Success-Stability Continuation

V136 was stopped after 697 fresh updates at true step `11779538944`. Its best
early window already met the original V135 Stage-43 hits/length pair, but the
last 48 updates fell to about `11.51` hits, `0.891` length fraction and `0.923`
view. Meanwhile the KL to the old V123 anchor decreased, so continuing the
same optimizer was pulling an adapted policy toward an obsolete actor rather
than exposing a new DR or capacity limit.

V137 profile
`goal_d455_sport_taskspace_record_new3_sim2real_success_stability_v137`
preserves Stages 1--42 exactly and restores every V135 graduation gate. From
Stage 43 onward it changes only the completion event from 20 points after ten
hits to 30 points after twelve hits. The selected Stage-43 parent already
spent 5400 updates in the identical domain, so Stage 43 uses `min_updates=16`
and the unchanged 16-update strict hold; all later stage minimums and holds
remain exact. DR, control, observation, model, physics and every other reward
remain unchanged.

Four matched 512-environment diagnostics are under
`outputs/rl_sim/v137_stage43_parameter_tuning_20260828/`. A stage-local anchor
alone peaked at only `11.8409/0.91128` hits/length. Adding the success-aligned
completion reward at the old PPO scale reached `11.9733/0.92808` but did not
clear hits. The selected LR `1.5e-5`, clip `0.06`, target KL `0.002`, log-std
`[-4.2,-3.8]`, source-local anchor coefficient `0.01` combination produced a
16-update complete strict pass at updates 22--37 and peaked at
hits/length/full/view `12.4615/0.95892/0.90769/0.95366`, with qacc exceedance
`0.00259` and exact KL `0.00104`. A repeat was weaker; it is retained as
variance evidence and no bounded actor is promoted.

The formal launcher is
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v137_success_stability_from_v135_stage43.sh`.
It restores the V135 update-5400 checkpoint at true step `11688181760`, Adam
time `1152135`, SHA-256
`67c731bb518b051910508bfd3d3a8a37743eaadc309edd46cb3fd6105b23bed7`,
and uses the same checkpoint as the resumed Stage-43 actor anchor. Future
stage transitions create fresh local anchors. Formal training uses 1024
environments, 128 steps, minibatch 16384, two epochs, no update cap and block
validation in tmux `pp_gpu1`. It continues W&B-online ID `v130g1r1` with
`resume=must` and W&B-only offset `852623360`, placing the first V137 point
one 131072-step rollout after the final V136 point without altering the true
optimizer step. V137 remains simulation-only and inherits the V135 deployment
blockers.

## GPU1 V138 Failure-Time Survival Continuation

V137 Stage 43 was stopped after update 1293. Across 1292 recorded leaves,
rolling hits reached 12 on 130 updates, but rolling length peaked at only
`0.91642` and never reached the unchanged `0.92` gate. The closest joint point
was update 366 at hits/length/full/view
`12.09656/0.91642/0.82829/0.94220`. The last 48 raw updates averaged
`11.89269` hits, length fraction `0.90579`, full `0.80587` and view `0.93528`.
Last-64 PPO exact KL was `0.000879`, explained variance `0.96839` and qacc
exceedance `0.00320`; the long plateau is therefore not optimizer instability,
a new DR transition or exhausted model capacity.

The successor profile is
`goal_d455_sport_taskspace_record_new3_sim2real_failure_time_v138`. Stages
1--42 are exact V137 copies. From Stage 43 onward the only change is
`early_termination_remaining_horizon_penalty=18`, emitted as
`-18*(1200-failure_step)/1200` on a true termination and zero on time-limit
truncation. The bound equals the full-horizon value of the existing
post-first-hit alive signal (`3 reward/s * 0.005 s * 1200`), following the
accepted GPU0 V85 causal failure-time repair. The V137 30-point completion
event gated by 12 hits, every other reward, every gate, PPO, control,
observation, physics and DR remain unchanged.

The bounded 512-environment direction trial uses
`run_gpu1_record_new3_v138_failure_time_trial.sh`, at most 64 updates and
offline W&B. The formal launcher is
`run_gpu1_record_new3_v138_failure_time_from_v137_best.sh`. Both restart from
the V137 best checkpoint at Stage-43 update 483, true step `11751489536`, Adam
`t=1159863`, SHA-256
`c1c4217447445f21393f981a497125050c6f3b61970bd3375efbcbcef2843499`;
the trial actor is never promoted. Formal training uses 1024 environments,
128 rollout steps, minibatch 16384, two epochs, LR `2.0e-5`, clip `0.06`,
target KL `0.002`, log-std `[-4.2,-3.8]`, stage-local anchor coefficient
`0.01`, no stage cap and block validation. It continues W&B-online run
`v130g1r1` with `resume=must` and W&B-only offset `958922752`. V138 remains
simulation-only and inherits every V135 deployment blocker.

The selected bounded comparison used the same V137 parent and V138 reward.
At LR `1.5e-5`, the last 48 updates were hits/length/full/view
`12.0372/0.91210/0.82525/0.94080` with declining length. LR `2.0e-5`
improved these to `12.1314/0.91732/0.82742/0.94311`; final rolling
hits/length were `12.06497/0.91555`, exact KL `0.00127`, and qacc exceedance
`0.00307`. Formal training therefore uses `2.0e-5` but still restarts from the
immutable V137 checkpoint, never a trial actor. V139's otherwise identical
30-point failure-time trial was rejected at last-48
`11.8454/0.90728/0.81309`; it is negative evidence, not a formal successor.
