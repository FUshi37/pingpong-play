# GPU1 V153 Training Companion

This directory is a frozen copy of the exact training source and evidence
contracts used by the selected GPU1 V153 checkpoint.  It is training/audit
material, not a real-robot launcher and not a replacement for the active
shared trainer.

## Selected source identity

The selected model was produced by the signal-resumed formal run:

```text
pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/
formal_gpu1_v153_dual_domain_resume1_stage57_u1_nsteps128_online1_20260901
```

Its `source_snapshot/` files are byte-identical to the companion source files
listed below:

| File | SHA-256 |
| --- | --- |
| `train_juggle_mjx_curriculum.py` | `390d76daf5a8b1bc37237a3389d945ac29e10372a25d4f5da1a80eaf23b8a3d3` |
| `train_juggle_mjx_ppo.py` | `8a0bebb21c8613676095e52c26e33a7c04d90528eb60dae54edf8c18901380d3` |
| `mjx_juggle_env.py` | `254eb1a94a1286d7c41fa5e3d9548dfdef41f47bd2d3fe5ea639483f41bfb7c4` |
| `mjx_smoke.py` | `fd0c4748039ebb53e022857f3cbf673ef50f0ce2ee910655742bb57ad5ce8f68` |
| `ball_mass_measurement.py` | `c298615d811666e1d1a8702ede56bef49482b8bc9f5eab408c083e3e45de7b88` |
| `run_with_host_memory_guard.sh` | `82b38cf6845501e58b0d37a6c98e34b6dc04f99eaedbc1c6c6e226054e3b8470` |
| `moz1_pd.xml` | `7d98f2adfdbad6082be0defcec2dbd0cbbcaf1f0fc06ce45ba424b5b3257cc92` |
| `meshes/` tree | `f5046e260ae4293675214bb0d9b46bed0ea5c3b5483b7377018a1ae114222851` |

The selected checkpoint identity and deployment boundary are documented in
`pingpong_controller/tools/rl_2real/GPU1_V153_REAL_ROBOT_DEPLOYMENT.md`.

## V153-specific training changes

V153 is continuation-only.  It restores the completed V152 B2 actor, 279-D
critic, Adam moments, immutable V146 actor anchor and 8192 mature-main replay
observations.  It replaces the coarse B2→B3 bridge with reset-only fractions:

```text
1, 2.5, 5, 7.5, 10, 15, 20, 25, 35, 50, 65, 80, 100 percent
```

Only five falling-contact reset supports interpolate: time-to-contact,
incoming apex, incoming XY speed, contact XY jitter and contact local-Y.
Observation, QACC action, command-state integration, reward, contact/physics,
fixed base, measured `[3.9,4.1] g` ball mass DR and PPO settings remain fixed.

Graduation requires independent frozen recovery and unchanged-main retention
validators.  The final proof checks exact B3 and the unchanged main task rather
than accepting an aggregate mixture.  Actor replay KL is bounded at `0.025`.
This dual validator is opt-in and does not alter historical profiles.

The selected checkpoint is the best checkpoint available when training was
safely stopped: Stage 60/66, label `p025`, meaning 25% (not 2.5%).  It has not
passed exact B3 or the final combined proof.

## Launchers and immutable inputs

`run_v153_initial_from_v152_stage51.sh` records the
initial V152→V153 transition.  It is content-addressed to the original trainer
snapshot (`59ae7234...`) and intentionally refuses the later resume-capable
shared trainer.  Exact replay of that first process must use its frozen source
snapshot.

`run_v153_resume1_selected_source.sh` is the launcher that matches
the selected model's source snapshot.  It pins:

- interrupted source checkpoint SHA-256
  `6494393731b00a7c2fe96b2051e87e3b15c2ecf30368b5d38fe008ace478adcc`;
- source `curriculum_progress.csv` SHA-256
  `4e8ad893e3c99be3f0c57e29617100cf9fcae9113f38986fd819dc2a5c8b0df2`;
- V146 actor anchor and mature-main replay hashes;
- frontier and bounded dual-domain acceptance hashes;
- measured ball-mass manifest, GPU UUID, W&B ID/offset, source files and
  no-preallocation/host-memory/temperature gates.

Parent/interrupted checkpoints, the 8192-row replay array and W&B histories are
run artifacts, not source code; they are deliberately not duplicated in this
companion commit.  Their paths and hashes remain in the launchers.  The small
hashed mass/frontier/bounded acceptance JSON files are included because the
launchers treat them as fail-closed evidence contracts.

The launchers intentionally retain the original absolute experiment paths and
content hashes.  They are evidence of the actual commands, not portable
one-click launchers.  Missing parent artifacts, another GPU UUID, another W&B
lineage or a changed source hash must continue to fail closed.

## Shared-code scope

The trainer and environment preserve every historical profile and include the
V143→V153 fixed-base/4 g/falling-reset lineage.  They also contain adjacent
opt-in GPU0/GPU1 profiles required for old checkpoint loading and profile
construction.  This is the exact code the selected V153 process ran; it should
not be reduced to a copied standalone V153 function that silently loses those
compatibility contracts.

`verify_release.py` checks the local source/evidence hashes, the selected
stage identity and launcher syntax without importing JAX or starting training.
The formal development tests additionally covered legacy floating-base
invariance, aligned/fixed base, 4 g mass/inertia sampling, mixed/falling resets,
V143→V153 profile monotonicity, dual-domain gates, exact signal-resume state,
PPO continuation behavior and host-memory SIGINT forwarding.
