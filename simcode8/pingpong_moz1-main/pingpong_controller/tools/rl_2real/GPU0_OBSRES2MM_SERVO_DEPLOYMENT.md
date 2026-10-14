# GPU0 2 mm observation residual + servo planner deployment and commit report

## 1. Submission identity

- Branch: `main`.
- Direct parent: `7cf280441c0ba93cf13ebca04887037ffb9d1faf`
  (`deploy: add GPU1 resume8 inverse-MPC policy`, 2026-07-27).
- Earlier comparison point: `f2c08ae6f1ba23572583bc1ffad25bd205c48da0`
  (`idealpd sim2real & ideal view & ideal range`, 2026-07-22).
- W&B run: `2w5tw9fb`, project `fushi37/pingpong-mjx`.
- Frozen model:
  `pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/mjx_curriculum_best.pkl`.
- Model SHA-256:
  `ed495f4445c21b4af7fd1acba420ba3e76e68a86dfdf2b35a7201acd641559c4`.

The checkpoint is stage 21
`launch19_final_measured_obsres2mm_servo_consolidation`, stage update 295,
global update 475, global step `5585240064`, and best stage score
`40.792066500571885`. Actor input/output is 67D/7D; the asymmetric critic is
231D and is training-only.

## 2. Result assessment

The training completed normally and passed the strict final-course gate at
stage update 375 (global update 555). The terminal 24-update window was:

| Metric | Terminal window | Final target |
|---|---:|---:|
| mean hits | 13.690 | 13.0 |
| mean length fraction | 0.968 | 0.95 |
| 1200-step episode rate | 0.929 | 0.86 |
| hit1 rate | 0.999 | 0.98 |
| hit3 rate | 0.992 | 0.90 |
| hit12 rate | 0.831 | 0.76 |
| hit-camera-visible rate | 0.9996 | 0.91 |
| lower-band hit rate | 0.983 | 0.80 |
| next-contact anchor error | 0.121326 m | <= 0.1215 m |

The frozen model is the run's task-score best checkpoint at update 295, not
the terminal checkpoint. A matched 64-episode deterministic replay at seed
20260729 compared both after training ended:

| Checkpoint | Hits | Mean steps | 1200-step rate |
|---|---:|---:|---:|
| task-score best, update 295 (submitted) | 13.92 | 1174.7 | 61/64 (95.3%) |
| strict-gate terminal, update 375 | 13.50 | 1168.8 | 61/64 (95.3%) |

Thus update 375 supplies the formal curriculum-convergence certificate while
update 295 is submitted because it is stronger in independent replay. This
selection is based on task performance, not checkpoint recency.

The submitted deterministic video at seed 20260729 completes all 1200 steps
with 12 hits, full rate 1.0, hit-camera and hit-band rate 1.0, and only normal
horizon truncation. The video is 1280x720, 30 fps, and 6.03 s.
The submitted `action_plot.png` covers the same complete episode and includes
policy action, normalized acceleration, normalized velocity, commanded/applied/
actual joint angle, and ball/racket height curves; its SHA-256 is
`8231d109f9186ea1dac4d8e67a02d22d83be95ecc521011fdc4470117f8e621a`.

## 3. Root-cause repair relative to the previous GPU0 plateau

A frozen-checkpoint, matched-seed 128-episode comparison isolated the old
launch17 plateau:

| Domain | Hits | Mean steps | Length fraction | 1200-step episodes |
|---|---:|---:|---:|---:|
| old synthetic frame DR | 12.40 | 1040.7 | 0.867 | 96/128 (75.0%) |
| measured 2 mm residual | 13.41 | 1105.7 | 0.921 | 107/128 (83.6%) |

The old course combined per-refresh noise with episode-fixed camera rotation,
velocity bias, and scale error larger than the measured real observation
residual. This was a hidden observation-domain mismatch, not missing-ball
handling and not insufficient task range. The new profile therefore changes
the observation distribution, not the graduation threshold:

- position refresh noise and position bias envelope: 2 mm;
- velocity refresh noise: 0.07 m/s;
- synthetic frame rotation, velocity bias, and scale distortion: disabled;
- missing/dropout: disabled for this training branch;
- no additional state estimator is used.

Launch17 still requires hits >= 12, length fraction >= 0.90, and full rate
>= 0.75. The final course still uses the original widest physical/reward
domain and requires hits >= 13, length fraction >= 0.95, and full rate >= 0.86.

## 4. Control and safety contract

This model is not the GPU1 no-planner method. Its fixed execution stack is:

1. fitted actuator and 72 ms delay conditioning;
2. actual-feedback inverse MPC;
3. target-aware servo trajectory planner;
4. unchanged XML position PD.

| Parameter | Value |
|---|---:|
| control rate | 200 Hz |
| compensation | inverse MPC |
| MPC feedback | actual joint state |
| beta / delay scale / tau scale | 1.2 / 1.05 / 0.75 |
| MPC horizon | 6 steps |
| max MPC delta | 30 deg |
| servo target planner | enabled |
| servo velocity scale | 1.0 |
| servo acceleration scale | 0.8 |

At the frozen update, servo planning was active on about 93.7% of samples and
servo safety feasibility was 1.0. The planner is therefore part of the learned
plant and must not be disabled at deployment.

`MJXPolicyController` already reads the checkpoint planner contract but
returns position goals without duplicating the simulated FOPDT/delay/planner.
The physical drive is expected to own the acceleration-integrated target
trajectory planning and enforce the declared position/velocity/acceleration
limits. If the robot drive does not provide that planner, implement and replay-
validate the same causal planner before running this model; do not deploy it
through the GPU1 no-planner path.

## 5. PPO change relative to the plateau run

The plateau used LR 1e-5, clip 0.15, target KL 0.006, actor-anchor coefficient
0.01, and failure-tail weighting. Its KL/clip activity was too low while
anchor and tail penalties opposed policy movement. The release run resets
optimizer state and uses:

- learning rate `5e-5`;
- clip range `0.2`;
- target KL `0.012`;
- two epochs, 1024 environments, 256 rollout steps;
- gamma `0.9995`, GAE lambda `0.99`, time-limit bootstrap;
- actor anchoring and failure-focused weighting disabled.

The frozen update had exact PPO KL 0.01175, clip fraction 0.174, explained
variance 0.933, no KL rollback, and no actor-anchor/failure-focus contribution.
Network structure and control method are unchanged.

## 6. Code delta relative to the previous two commits

Relative to direct parent `7cf28044` (GPU1 resume8), this submission adds:

- measured-residual launch17 and final curriculum profile while retaining the
  original widest physical/reward distribution and strict 1200-step gates;
- launch17 diagnostic/bridge profiles and hard-tail DR support used to isolate
  the plateau rather than hiding it by lowering thresholds;
- intercept/contact/recoverability metrics and episode-correct validator
  aggregation used for causal miss analysis;
- actual-feedback inverse-MPC plus servo-planner GPU0 checkpoint, exact launch
  script, deterministic video, action/observation traces, and deployment report.

Relative to `f2c08ae6` (ideal-PD67), this also inherits the direct parent's
fitted actuator, delay-conditioned 67D observation, inverse-MPC training and
deployment support, time-limit bootstrap, strict final metrics, and sim2real
validation stack. This commit does not modify network structure.

The dirty worktree also contains calibration JSON, XML experiments, random-
environment tools, residual-policy code, jerk/governor experiments, caches,
and unrelated outputs. They are intentionally excluded. In particular, the
current uncommitted residual actor additions in `mjx_policy_controller.py` and
jerk-governor additions in `safety_limiter.py` are not required by this
checkpoint and are not part of this submission.

## 7. Submitted files

- `pingpong_controller/tools/rl_sim/mjx_juggle_env.py`
- `pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py`
- `pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py`
- `pingpong_controller/tools/rl_sim/run_gpu0_launch17_obsres2mm_servo_v5.sh`
- `test/test_goal_d455_curriculum.py`
- `test/test_delay_conditioned_control.py`
- this document;
- exact W&B run config `wandb_config.yaml`;
- frozen model and video directory under
  `outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/`.

## 8. Verification status

- 64-episode matched deterministic selection replay: passed; submitted model
  has 13.92 mean hits and 95.3% full-episode rate.
- GPU deterministic video replay: passed, 12 hits and 1200/1200.
- Python syntax compilation: passed.
- curriculum configuration/invariant assertions: passed.
- shell syntax for the exact training script: passed.
- checkpoint one-step real-controller load/inference: must be run before
  physical deployment and is included below.
- full pytest is not claimed: the pingpong conda environment has JAX/MuJoCo
  but no pytest, while the system pytest environment lacks compatible JAX.

## 9. Exact submission commands

The pre-existing local commit was amended after training finished. The final
model, config, video, action plot, traces, and this report are already in
`HEAD`; unrelated dirty-worktree files remain unstaged. Verify and push with:

```bash
cd /home/yangzhe/Project/pingpong_controller
git branch --show-current
git log -1 --oneline --decorate
git diff --cached --quiet
sha256sum \
  pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/mjx_curriculum_best.pkl \
  pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/video/launch19_final_best_u295_seed20260729.mp4
git push origin main
```

## 10. Pre-deployment controller smoke test

```bash
PYTHONPATH=/home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim \
/home/yangzhe/miniconda3/envs/pingpong/bin/python \
  pingpong_controller/tools/rl_2real/mjx_policy_controller.py \
  --checkpoint pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/mjx_curriculum_best.pkl \
  --robot-xml pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --steps 1 --ball-valid \
  --arm-q-deg 32,-58,43,98,26,-6,47 \
  --arm-dq-deg-s 0,0,0,0,0,0,0
```

The summary must report `obs_dim=67`, `comp_mode=inverse_mpc`, actual
feedback, `drive_target_tracking_planner=True`, velocity scale 1.0, and
acceleration scale 0.8 before physical testing.
