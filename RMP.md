---
name: rmp-replay-conclusions
description: RMP replay conclusions and parameter settings
metadata:
  type: project
---

# RMP replay conclusions and settings

## What we learned
- The old Markdown summary showed `n/a` for racket fields because it was reading pre-update replay reports.
- Racket center metrics must be captured from each replay step, not reconstructed from the final MJX state.
- The replay report should carry a full `racket_metrics` block, and the batch summary should read that schema directly.
- Parallel batch replay is useful here because each recording can run in its own subprocess with a bounded worker pool.

## Current replay schema goals
- Racket center position
- Racket center orientation
- Racket center linear velocity
- Racket center linear acceleration
- Racket center angular velocity
- Racket center angular acceleration

## Current batch settings
- Use `--workers` to run multiple replay subprocesses in parallel.
- Keep workers bounded so MJX/JAX memory use stays controlled.
- Keep per-recording output directories separate.

## Vectorized batch replay entrypoint (current)
The canonical batch replay is the vectorized MJX entrypoint in simcode5. Each
recording is one MJX environment lane; waves of `--batch-size` lanes run in one
compiled pass:

```bash
cd simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim
conda activate pingpong_mjx
JAX_PLATFORMS=cpu python batch_rmp_replay_mjx.py \
  --model <workspace>/src/pingpong_moz1/pingpong_controller/tools/rl_sim/moz1_rmp_equiv.xml \
  --record-root <workspace>/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings \
  --include-group record_new3 --include-group record_new4 --include-group record_new5 \
  --output-root <output-dir> --batch-size 8 \
  --calibration-profile recovered_yaml_pd_fit \
  <per-joint refine baseline parameters, see below>
```

This entrypoint replaces the older subprocess-based
`src/.../rl_sim/batch_rmp_replay_validation.py` runs. It writes a full
`racket_metrics` block per report and renders `batch_summary.md` itself. Use
`--rerun` to recompute existing per-record artifacts. The current tuned
parameter set (refine baseline) is:

```text
--controller-kp-scale-per-joint 3.509999990463257,5.760000228881836,4.949999809265137,3.6600000858306885,2.3399999141693115,4.139999866485596,3.509999990463257
--controller-kd-scale-per-joint 4.52400016784668,5.335999965667725,6.380000114440918,4.52400016784668,7.424000263214111,10.0,5.800000190734863
--rmp-jnt-kp-scale-per-joint 1.0,1.0,1.0,1.0,1.0,1.0,1.0
--rmp-jnt-kd-scale-per-joint 1.0,1.0,0.9200000166893005,1.0,1.0,1.0,1.0
--dof-damping-scale-per-joint 0.6083999872207642,0.9983999729156494,0.9983999729156494,1.9199999570846558,0.5799999833106995,1.1699999570846558,0.7799999713897705
--dof-armature-scale-per-joint 0.45239999890327454,1.649999976158142,0.9200000166893005,1.2200000286102295,0.5799999833106995,0.5335999727249146,0.5799999833106995
--rmp-output-delay-steps-per-joint 18,19,19,18,18,18,18
```

## Vectorized batch replay results (2026-08-20)

Full record_new3/4/5 run with the refine-baseline parameters above, one MJX
environment lane per recording, waves of 8 lanes, CPU backend. Output tree:
`analysis_results/rmp_batch_validation_record_new3_4_5_parallel/`.

Status: 64/65 passed. The single failure is
`record_new4/probe_recording_20260811_205641`, which is skipped in preflight
because its `control_samples.jsonl` contains no published seven-joint samples
(a data problem, not a replay problem).

Group means (joint error vs real recording):

| group | n | q RMS rad | dq RMS rad/s | qdd RMS rad/s² | RMP q RMS rad | max q err rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| record_new3 | 16 | 0.009427 | 0.136779 | 3.8798 | 0.018113 | 0.185798 |
| record_new4 | 25 | 0.012721 | 0.155615 | 4.1559 | 0.021009 | 0.995954 |
| record_new5 | 23 | 0.025193 | 0.259184 | 5.8253 | 0.026630 | 1.234861 |

Racket end-effector sim-vs-real error (2026-08-20, computed on the same
64 passing records; `right_ee_site`, real side = MuJoCo FK on recorded joint
angles + raw `ros.jointVelocity`, sim side = FK on MJX replay q/qd, both on
the replay time grid, SG window 15 order 3 on tip velocity for accelerations,
gaps not bridged):

| group | pos RMS norm m | angle RMS norm deg | lin vel RMS norm m/s | ang vel RMS norm rad/s | lin acc RMS norm m/s² | ang acc RMS norm rad/s² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| record_new3 | 0.006 | 1.062 | 0.100 | 0.091 | 3.247 | 2.200 |
| record_new4 | 0.009 | 1.482 | 0.120 | 0.106 | 4.398 | 2.676 |
| record_new5 | 0.016 | 2.319 | 0.201 | 0.179 | 6.692 | 4.203 |

Across all 64 records the position RMS norm is mean 0.011 m / median 0.006 m
and the angle RMS norm mean 1.68° / median 0.82°. The tail comes from the same
two divergence cases seen in the joint metrics —
`record_new4/203833` (0.092 m, 17.4°) and `record_new5/143204` (0.152 m,
26.4°); excluding both, group position RMS norms are 0.0064/0.0056/0.0096 m
(new3/new4/new5) and angle RMS norms 1.06°/0.82°/1.23°. Millimeter-level
position accuracy is therefore not yet reached group-wide: new3/new4 are close
(5–6 mm), new5 is ~10 mm without the divergent records. Per-axis tables,
per-record details and the machine-readable aggregate live in
`analysis_results/racket_sim2real_rmp_parallel_20260820.json` (produced by
`src/.../rl_sim/analyze_racket_sim2real_batch.py`).

Notes on interpretation:

- Every passing report carries a full `racket_metrics` block (position,
  orientation quaternion, linear velocity/acceleration, angular
  velocity/acceleration in world and local frames). These are the raw replay
  traces, not errors; the sim-vs-real error numbers above are computed by
  `analyze_racket_sim2real_batch.py` from the stored joint traces.
- The two large `max q err` values come from single recordings —
  `record_new4/probe_recording_20260811_203833` (0.996 rad) and
  `record_new5/probe_recording_20260813_143204` (1.235 rad). These are the
  known divergence cases to inspect before claiming hardware equivalence; the
  RMS values are not dominated by them.
- Segfaults occur sporadically inside `discover_recordings` (`pathlib.glob`,
  pre-JAX, seen only in the full 3-group run). Retrying resumes safely because
  the script reuses completed per-record reports. Batch-size 8 per wave is
  stable; the earlier batch-size 16 attempt died in wave 1.

## Replay pipeline notes
- `record_new3`, `record_new4`, and `record_new5` are the primary validation groups.
- Markdown summaries should be generated from the replay JSON reports after rerun.
- Failed samples should be inspected from each replay log before interpreting their metrics.
- RMP and XML PD parameters are maintained in a single JSON file:
  `recovered_rmp/configs/rmp_equiv_refine_baseline.json` (schema:
  `recovered_rmp/configs/rmp_config_schema.json`). The convenience API is
  `recovered_rmp/interface.py` (`load_rmp_config` / `create_planner` /
  `drive_joint_rmp`). When retuning, update the JSON and both the replay
  batch command lines and the interface stay in sync.

## Why this matters
The racket metrics and replay settings are part of the calibration/validation contract for the MJX/RMP pipeline.

## How to apply
- Rerun the relevant recording groups with the updated replay script.
- Regenerate Markdown summaries from the new JSON reports.
- Review any failed replay logs before using the batch statistics.
- Keep this note updated if the schema or worker settings change.

Related: [[rmp-replay-status]]
