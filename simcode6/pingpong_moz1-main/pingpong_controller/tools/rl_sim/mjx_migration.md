# MJX/JAX GPU Training Migration

This project currently trains with Stable-Baselines3 PPO on a Gymnasium env that
steps MuJoCo through the normal Python/C API. Setting `--device cuda` in that
pipeline only moves the PPO neural network to the GPU; the MuJoCo physics and
environment logic still run on CPU.

To get real MuJoCo GPU acceleration, the environment must be ported to MJX/JAX:
batched state lives on the accelerator, `reset` and `step` are JAX functions,
and rollout collection is compiled with `jax.jit`/`jax.vmap`.

## 0. Equivalence Target

The MJX/JAX path is now a full training path, but it still cannot be guaranteed
to be bit-for-bit identical to the
MuJoCo-CPU/SB3 path. The reasons are structural:

- MJX-JAX and MuJoCo CPU use different physics implementations and numerical
  execution paths.
- The full CPU XML contains collision pairs that MJX-JAX does not implement.
  The MJX training XML therefore keeps visual meshes non-colliding and adds
  MJX-safe primitive contact proxies for ball/racket, non-racket body contact,
  and base/floor support.
- SB3 PPO and this JAX PPO use different random number generators, optimizer
  implementations, batching order, and floating-point reductions.
- GPU execution is massively parallel, so reduction order and stochastic rollout
  ordering differ from CPU SB3.

The practical target is therefore:

- same action interface
- same observation layout
- same curriculum arguments where implemented
- reward and termination logic matched by explicit parity tests
- statistically comparable validation curves under fixed evaluation seeds

Current implementation status:

- `MjxJuggleConfig` mirrors every field in CPU `JuggleConfig`.
- Reward terms include cadence/cap, fast-hit gating, camera penalties,
  contact flatness, sticky contact, non-racket contact proxies, arm limits,
  post-hit shaping, and the existing center/base/chest constraints.
- Observation noise, observation rate, dropout, action latency, and observation
  latency run inside the JAX env state.
- Domain randomization now uses a batched MJX Model in `EnvState`: each parallel
  env can reset with its own ball mass/inertia, gravity, ball/racket friction,
  pair/geom solref, dof damping, dof armature, action scale, latency, dropout,
  observation noise, and racket mount position/rotation/radius offset.
- The remaining unavoidable differences are numerical/backend differences and
  the MJX-safe proxy collision XML, not missing per-env model mutation.

Use the parity guardrail before calling a migrated stage equivalent:

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim
JAX_PLATFORMS=cpu python compare_mjx_cpu_env.py --steps 200
```

For a stricter regression check:

```bash
JAX_PLATFORMS=cpu python compare_mjx_cpu_env.py --steps 200 --strict
```

If this test diverges, fix environment parity before interpreting training
curves.

## 1. Install GPU JAX and MJX

Use the existing `pingpong` env after the NVIDIA driver is working.

```bash
conda activate pingpong
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda12]" mujoco-mjx
```

Verify that JAX can see the GPU:

```bash
python -c "import jax; print(jax.devices())"
```

Expected output should include an NVIDIA GPU device, not only `CpuDevice`.

## 2. Smoke-test this robot XML in MJX

From this directory:

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim
python mjx_smoke.py --n-envs 1024 --steps 200
```

This script reuses the same ball/racket XML patching path as
`rl_juggle_env_random.py`, then writes an MJX-friendly temporary XML. The XML
keeps visual geometry for rendering, disables unsupported broad collisions, and
adds primitive contact proxies for the task-relevant contacts. The full robot
scene contains broad collision pairs such as `cylinder`/`box`, which MJX-JAX
does not implement.

The script transfers the compiled MuJoCo model to MJX, batches the data, and
runs a JIT-compiled rollout. Passing this test means the model is at least
viable for an MJX port.

For quick debugging, start smaller:

```bash
python mjx_smoke.py --n-envs 16 --steps 20
```

To intentionally test the original full collision masks:

```bash
python mjx_smoke.py --n-envs 16 --steps 20 --full-collision
```

This is expected to fail on MJX-JAX unless the XML collision geoms are simplified.

## 3. Curriculum Migration

`train_juggle_mjx_curriculum.py` now runs the whole curriculum in one process.
It keeps Stage 1a-3b in no-DR mode to match the CPU commands, then enables the
Stage 4 DR presets progressively. The stage gate can be convergence-based or
fixed-step:

```bash
python train_juggle_mjx_curriculum.py \
  --n-envs 1024 \
  --n-steps 64 \
  --minibatch-size 8192 \
  --update-epochs 4 \
  --advance-mode converged \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_curriculum
```

## 4. Batched Randomized MJX Model

`mjx_juggle_env.py` keeps the model itself in `EnvState`, not only `mjx.Data`.
On reset, the environment samples per-env DR values and builds a batched model
pytree. Rollout then calls:

```python
jax.vmap(lambda model, data: mjx.step(model, data))(state.model, state.data)
```

That makes contact/actuator/racket-mount DR physical in MJX rather than only
logging sampled values. `reset_done` replaces the model for finished envs only,
so each new episode receives fresh model randomization while unfinished envs
continue with their current parameters.

The randomized model fields are:

- `opt.gravity`
- `body_mass` and ball `body_inertia`
- `geom_friction`, `geom_solref`
- ball/racket explicit `pair_friction`, `pair_solref`
- `dof_damping`, `dof_armature`
- racket `body_pos`, `body_quat`, and racket wood/rubber `geom_size`

The current `JuggleEnv` uses Python objects, NumPy mutation, MuJoCo `MjData`,
Python loops over contacts, dictionaries, and Gym wrappers. Those patterns are
fine for SB3, but they cannot be compiled efficiently by JAX.

## 5. Training stack

SB3 should not be the training loop for the MJX path. It expects a CPU-style Gym
environment and will copy observations/actions across the Python boundary.

Use one of these instead:

- a small JAX PPO loop written for this env
- Brax-style PPO components
- a MuJoCo Playground style training loop

Keep the existing SB3 scripts as the reference implementation while validating
the MJX port.

## 6. Run the Stage 1a JAX PPO prototype

The first MJX training path is:

```text
mjx_juggle_env.py
train_juggle_mjx_ppo.py
```

It implements the same 50-dimensional observation layout and 7-dimensional
right-arm acceleration action interface used by the CPU reference.

Start with a short run:

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim
python train_juggle_mjx_ppo.py \
  --n-envs 1024 \
  --n-steps 64 \
  --total-steps 1000000 \
  --minibatch-size 8192 \
  --update-epochs 4 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_stage1a
```

Log directly to Weights & Biases:

```bash
python -m pip install wandb
wandb login

python train_juggle_mjx_ppo.py \
  --n-envs 1024 \
  --n-steps 64 \
  --total-steps 1000000 \
  --minibatch-size 8192 \
  --update-epochs 4 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_stage1a \
  --wandb \
  --wandb-project pingpong-mjx \
  --wandb-name stage1a-mjx-1024env
```

If GPU memory is tight, reduce `--n-envs` to 512 or 256 first. The first update
includes JIT compilation, so it is much slower than later updates.

This prototype uses one JAX device by default. Select a specific GPU with:

```bash
CUDA_VISIBLE_DEVICES=0 python train_juggle_mjx_ppo.py --n-envs 1024 --n-steps 64
```

True multi-GPU training requires a later `pmap`/sharding pass.

## 7. Run The MJX-Compatible Curriculum

Use this entry point when you want one process to walk through the named
curriculum stages and keep the same policy/optimizer across stages:

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim
python train_juggle_mjx_curriculum.py \
  --n-envs 1024 \
  --n-steps 64 \
  --minibatch-size 8192 \
  --update-epochs 4 \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_curriculum \
  --wandb \
  --wandb-project pingpong-mjx \
  --wandb-name mjx-curriculum
```

By default this runner now uses convergence-gated advancement. A stage keeps
training until a moving window of recent updates satisfies that stage's
`target_mean_hits` and `target_mean_len_frac`. The original stage step counts
are left as curriculum reference values; they do not stop training in
`--advance-mode converged`.

Important flags:

- `--advance-mode converged`: default; advance only after the stage target is met
- `--advance-mode fixed`: old behavior; advance after the configured step budget
- `--convergence-window 5`: number of recent updates used for the moving average
- `--convergence-min-episodes 32`: ignore sparse updates with too few completed episodes
- `--max-stage-updates 0`: optional safety cap per stage; `0` means no cap
- `--allow-unconverged-advance`: continue anyway if the safety cap is exhausted

Smoke test only the first stage:

```bash
python train_juggle_mjx_curriculum.py \
  --n-envs 16 \
  --n-steps 8 \
  --stage-steps 128 \
  --max-stages 1 \
  --advance-mode fixed \
  --save-dir /tmp/mjx_curriculum_smoke
```

This curriculum runner is separate from `train_juggle_mjx_ppo.py`. It uses the
same stage names as `training.md`, but only changes knobs currently implemented
in `MjxJuggleConfig`. CPU-only features that are not yet implemented in MJX are
logged as stage notes instead of being silently faked:

- camera visibility penalty
- ball observation noise/dropout
- latency randomization
- full contact/actuator/racket-mount domain randomization

The MJX reward now includes the CPU reward terms that most affect arm posture
and chest-front behavior: racket/chest penalties, ball anchor/base penalties,
post-hit survival/intercept shaping, racket upward-drift penalty, and actual
arm velocity/acceleration-limit penalties. Stage configs mirror the implemented
numeric CLI settings from `training.md`; the remaining differences are the
features listed above and the PPO implementation itself.

## 8. Validate a JAX PPO checkpoint

Headless validation:

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim
python validate_juggle_mjx_ppo.py \
  --checkpoint /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_stage1a/mjx_ppo_last.pkl \
  --episodes 20 \
  --n-envs 32 \
  --deterministic
```

Live viewer for the first environment:

```bash
python validate_juggle_mjx_ppo.py \
  --checkpoint /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_stage1a/mjx_ppo_last.pkl \
  --episodes 3 \
  --n-envs 1 \
  --deterministic \
  --render \
  --realtime
```

Validation writes per-episode results to `validation.csv` next to the checkpoint
unless `--results-csv` or `--no-save-csv` is set. Each completed episode prints
its done reason, for example `ball_too_low`, `racket_too_low`, or `truncated`.

## 9. Episode Length And Early Termination

Current MJX episodes use:

```text
timestep = 0.001 s
frame_skip = 5
policy dt = 0.005 s
horizon_sec = 6.0 s
max episode steps = 1200 policy steps
```

A ball contact does not directly end the episode. Early termination happens if
one of these `done/*` terms is true:

- `done/ball_too_low`
- `done/ball_too_high`
- `done/ball_x_out_of_bounds`
- `done/ball_y_out_of_bounds`
- `done/racket_too_far_from_anchor`
- `done/racket_too_high`
- `done/racket_too_low`

These terms are now logged to CSV and W&B for both single-stage and curriculum
MJX training.

If validation completes one hit and then terminates, inspect the printed done
reason and the `reward/*` terms. In the current MJX env, racket Z-limit
termination receives an explicit penalty, and ball-miss termination after a hit
also receives a penalty. A checkpoint that still gets only one hit is not
considered converged for curriculum advancement.

## 10. Visualize Initialization

Original CPU/Gym environment:

```bash
python show_env.py --steps 2000
```

MJX Stage 1a environment:

```bash
python show_mjx_env.py --steps 2000
```

Print only the initial base, racket, ball, and right-arm joint state:

```bash
python show_mjx_env.py --headless --steps 0
```

The key sanity checks are:

- ball starts above the racket by roughly `ball_launch_height`
- `ball-racket` XY is near zero in fixed Stage 1a
- right arm starts near the target posture, without large joint-limit contact
- the racket site is visually centered on the red rubber face

## 11. Training Curves

MJX PPO writes a CSV file:

```text
outputs/rl_sim/logs_mjx_stage1a/progress.csv
```

The most useful columns are:

- `mean_return`
- `mean_hits`
- `sps`
- `loss`
- `policy_loss`
- `value_loss`
- `entropy`
- `approx_kl`
- `explained_var`
- `reward/*`
- `done/*`

`mean_return` is not bounded by a fixed theoretical maximum because it depends
on hit count, dense shaping, and termination penalties. As a practical guide,
values around `3-4` usually correspond to a single rewarded hit, not stable
juggling. Two clean hits should move the return clearly above that range, and
three or more hits should be much higher. For curriculum decisions, prefer
`mean_hits`, `mean_len`, and `done/*` over `mean_return` alone.

`policy_loss` in PPO can rise and then settle because the loss is measured
relative to the old policy and normalized advantages, not as a direct task score.
Use `approx_kl`, `clip_frac`, `value_loss`, `entropy`, `mean_hits`, `mean_len`,
and the reward/done breakdown to judge whether training is healthy.

For MJX curriculum runs, remember that `global_step` counts all parallel
environment steps. With `--n-envs 1024 --n-steps 64`, each PPO update adds
65,536 samples, but each individual env only contributes `64 * 0.005 = 0.32s`
of continuous experience before bootstrapping. That is a short horizon for
juggling: a clean hit-to-hit interval is usually around `0.30-0.65s`, and the
episode horizon is `1200` env steps (`6s`). The large batch stabilizes gradients,
but it does not by itself solve temporal credit assignment.

Recommended MJX PPO rollout settings:

- Early hit-discovery stages: `--n-steps 128` if GPU memory allows.
- Cadence/camera/DR stages: prefer `--n-steps 256`.
- Keep `--n-steps 64` only for smoke tests or when memory/compile time is the
  immediate bottleneck.

For example:

```bash
python train_juggle_mjx_curriculum.py \
  --n-envs 1024 \
  --n-steps 256 \
  --minibatch-size 16384 \
  --update-epochs 4 \
  --advance-mode converged \
  --save-dir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_curriculum_nsteps256
```

Also keep `gamma` aligned with the CPU SB3 baseline (`0.995`).

The original SB3 training script writes TensorBoard events under each stage's
`tb/` directory. Example:

```bash
tensorboard --logdir /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim
```

To upload existing SB3 TensorBoard logs to W&B:

```bash
python -m pip install wandb
wandb login
wandb sync /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_rl_juggle_random_stage1a/tb
```

To upload an existing MJX CSV after training:

```bash
wandb sync /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/logs_mjx_stage1a
```

## 12. Performance notes

Messages like `Failed to import warp: No module named 'warp'` are warnings about
the optional MJX-Warp backend. MJX-JAX can still run. To use Warp later:

```bash
python -m pip install --upgrade "mujoco-mjx[warp]"
```

For this task, contacts matter. If the full visual robot XML is slow or fails in
MJX, create a training XML with simplified collision geoms:

- racket collision as primitive cylinder/capsule/box
- ball as sphere
- nonessential visual meshes set to non-colliding
- only keep collision geoms that can actually affect the ball

That simplification often matters more than increasing `--n-envs`.
