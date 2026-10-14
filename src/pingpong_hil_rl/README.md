# ROS2 physical-simulator + FlashSAC training

This package uses the physical simulator as the **only arm plant**. MuJoCo keeps
the ball, racket geometry, collision and simcode4 reward/termination logic, but
the seven right-arm joints are overwritten at every 1 ms MuJoCo substep by an
interpolation of consecutive `/joint_states` samples.

The implementation intentionally has no learned/analytic actuator model, no
compensation, no artificial action/observation latency and no servo target
planner. Ball/contact/camera uncertainty remains randomized because the real arm
does not provide those dynamics.

## Data and timing contract

- Policy/control target: 200 Hz; `/joint_states` is consumed with RELIABLE QoS.
- Commands: `mc_core_interface/msg/MechUnitCmdArray`, live right-arm `mu_idx=3`, joint mode,
  RELIABLE depth 10 on `/mx_mix_command`; startup requires a matched subscriber.
- A command is followed by at least one strictly newer source-stamped state.
  Header stamps are used for order/deltas only, never compared with the local
  wall clock.
- `JointState.effort` is ignored. Joint acceleration is computed from adjacent
  velocity samples and normalized by simcode4 joint acceleration limits.
- Observation is simcode4's 50 floats plus 40 x 7 relative command history,
  seven normalized measured accelerations, source-dt scale and receive-age
  scale: **339 floats**.
- The 40-sample history covers 200 ms, exceeding the measured 100–120 ms plant
  phase lag.
- The command watchdog repeats the last command during learner/checkpoint
  activity. Stale state, excessive feedback velocity or >15 degree tracking
  error triggers a fail-closed stop and actual-position hold.

FlashSAC learning runs in a separate thread. The real-time path owns ROS,
MuJoCo and an actor-only inference snapshot on a high-priority CUDA stream;
every 50 learner updates, weights are synchronized under a short lock. This
prevents a GPU optimizer update from
freezing ball physics between physical state samples.

## Safety precondition

Training requires exclusive ownership of `/mx_mix_command`. The implementation
checks this before creating its publisher and refuses to start if another
publisher exists. Do not use `--allow-existing-command-publisher` during normal
operation.

At startup and after any episode ending more than 5 degrees from the D455 pose,
the plant is returned by a feedback-gated 10 degree/s ramp. The ramp must settle
within 0.25 degree and is allowed only inside the configured per-joint recovery
window. This tolerance only decides when reset is complete; policy observations
and safety checks retain the physical simulator's real tracking error.

## Run

Read-only measurement (never publishes commands):

```bash
source /opt/ros/humble/setup.bash
source /home/spirit-ai/ros_pkg/movax_interface/install/setup.bash
export PYTHONPATH="$PWD/src/pingpong_hil_rl:$PYTHONPATH"
python3 -m pingpong_hil_rl.live_probe --samples 2000
```

Start training after the old command publisher has exited normally:

```bash
cd /home/spirit-ai/pingpong_moz1
./run_hil_flashsac.sh
```

Run only the low-speed reset state machine:

```bash
./run_hil_flashsac.sh --reset-only
```

Resume both model/optimizer and replay:

```bash
./run_hil_flashsac.sh \
  --resume runs/hil_flashsac/<run>/step000100000 \
  --resume-buffer runs/hil_flashsac/<run>/step000100000
```

The launcher selects the existing `pingpong_mjx` Python 3.10 environment, adds
its cuDNN 9 library directory, sources ROS2 and the Movax message interface, and
adds the official checkout at `third_party/FlashSAC` to `PYTHONPATH`.

Checkpoints are written every 25k interactions. A restartable model plus replay
snapshot is forced exactly at the 10k warmup boundary; replay is then also
written every 100k. Each checkpoint contains `hil_metadata.json`, which records the full
configuration and the disabled synthetic-plant features. SIGINT/SIGTERM first
stops sampling, drains the learner queue, checkpoints, and leaves the plant at
an actual-position hold.

## Curriculum

The five stages retain simcode4's basic progression: near-racket interception,
first hit, recovery, multi-hit, then robust spawn/contact training. Action
acceleration scale grows from 0.18 to 0.90 and target rebound height grows from
0.12 m to 0.42 m. The first two stages add a measured-paddle scaffold for face
flatness and pre-contact upward velocity; it is zero after the first confirmed
hit and disabled entirely in later stages. At the measured D455 pose the racket
height is about 0.537 m, so launch height is +0.32 m; the old CPU default +0.24 m
would cross its z=0.8 m termination threshold immediately.

One million interactions has a hard lower bound of about 83 minutes at an ideal
200 Hz, but the measured end-to-end rate after learning starts is currently
roughly 55--60 interactions/s, implying about 4.6--5.1 hours plus large replay
snapshot overhead. This cannot create more real transitions than wall time.
The measured learner ceiling is 79.5 updates/s, so online
UTD=0.25 (about 50 updates/s) leaves control headroom and avoids an ever-growing
queue; the full replay remains available for later offline updates. A Dreamer
world model can later add imagined updates,
but it should be evaluated only after this real replay buffer exists; it does
not remove the need for causal state capture and substantially increases model
and tuning complexity on a single 24 GB GPU.

## Verification performed

- 2,000 live RELIABLE samples: source median 5.000072 ms, p99 5.003332 ms,
  max 5.007375 ms, zero non-monotonic samples and zero gaps above 7.5 ms.
- Source rate: 199.9971 Hz. Receive callbacks can batch under host scheduling
  (observed max receive gap 69.86 ms), so the fail-closed state timeout is 100 ms;
  this avoids false stops while still refusing to fabricate missing feedback.
- Low-speed reset converged from the controller default pose to the D455 target
  with 0.0763 degree maximum residual; after an active learner smoke run it
  reconverged within 0.2329 degree.
- Active command smoke: 2.743 degree joint span, 21.48 degree/s peak feedback,
  1.108 degree peak tracking error and 0.275 degree final tracking error.
- Official FlashSAC actor, distributional critic, replay buffer, reward
  normalization and temperature completed 84 finite CUDA updates in a 400-step
  feedback-loop smoke run.
- RTX3090 batch-512/339-dim microbenchmark: 12.58 ms/update (79.5 updates/s),
  66 MiB allocated and 234 MiB reserved, with `torch.compile` disabled for the
  steady-state comparison.
- The unmodified-reward control ran 50,570 interactions and wrote valid 10k,
  25k and 50k checkpoints. Actor, critic and temperature remained finite;
  critic loss fell from 4.49 to 1.89, but confirmed hits remained zero.
- Three first-hit curriculum trials ran 26,163, 20,449 and 25,527 interactions.
  The best trial learned paddle up-cos as high as 0.979 and produced occasional
  +0.172 m/s ball vertical velocity, but none produced a confirmed hit. Even a
  0.12 m first-stage target and 5x stronger scaffold remained at zero hits by
  25k. This is evidence that pure, from-scratch, single-environment FlashSAC is
  not currently sample-efficient enough for the first-hit discovery problem;
  falling critic loss alone must not be treated as task success.
- Across these runs, measured peak tracking error was 7.72 degrees, peak joint
  feedback speed was 114.49 degrees/s and source intervals in logged samples
  stayed at or below 10.005 ms. These remain below the configured 15 degree and
  340 degree/s fail-closed limits. The learner queue was normally 0--2 and
  drained replay-save bursts (observed maxima 58--130).
- Offline contact baseline at the D455 pose measured racket z=0.5373 m and
  vertical normal component 0.653. A falling ball contacts the inclined paddle
  but does not launch upward with zero action. Consequently zero confirmed hits
  in the first 25k interactions is a curriculum/exploration finding, not a
  broken ROS-to-MuJoCo collision path.
- Eight local tests cover interpolation endpoints, measured-arm injection,
  observation size, feedback-gated reset, workspace clipping, one environment
  step and HIL configuration invariants.
# Four-trajectory plant replacement experiment

`run_hil_replay_equivalence.sh` sequentially replays the legacy, simcode2,
simcode3 and simcode4 traces against the physical simulator, records ROS
feedback, then feeds exactly that feedback through the production HIL MuJoCo
state-injection path.  Results are written below
`pingpong_hil_rl/pingpong_hil_rl/outputs` by default.  The HIL reset places the
ball at the racket centre with a uniformly sampled 5--10 mm **surface** gap.
