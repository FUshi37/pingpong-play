# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project focus

This repository is a ROS 2 / Python / MuJoCo-MJX ping-pong robot stack. The current high-priority work is the recovered RMP implementation in [recovered_rmp/](recovered_rmp/): make it match real robot joint behavior as closely as possible for simulation training. The target equivalence is not only joint angle and velocity, but also joint acceleration and racket end-effector position/velocity/acceleration.

Real robot recordings live under [src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/](src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/) as `record_new3`, `record_new4`, `record_new5` and matching zip files. Some recordings used `dq_adapter`; when fitting or validating hardware equivalence, prefer the raw ROS-provided values (`ros.jointPosition`, `ros.jointVelocity`, `ros.jointAcceleration`, `ros.jointSampleStampS`) over adapter-derived fields.

## Environment and build commands

Most commands assume the repository root:

```bash
cd /home/spirit-ai/pingpong_moz1
```

Useful conda environments mentioned in project notes:

```bash
conda activate pingpong_mjx   # MJX/JAX simulation and training work
conda activate pingpong       # ROS/control/camera tools when MJX is not needed
```

ROS 2 setup/build:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select pingpong_controller
colcon build --packages-select pingpong_replay
source install/setup.bash
```

The top-level [install/](install/) tree is a built copy; after changing ROS packages, run `colcon build` and re-source `install/setup.bash` before using `ros2 run`.

## Test commands

Recovered RMP unit tests:

```bash
python3 -m unittest discover -s recovered_rmp/tests -v
python3 -m unittest recovered_rmp.tests.test_core.RecoveredRMPTests.test_stateful_joint_rmp_returns_consistent_q_qd_qdd -v
```

ROS/package tests can be run directly with pytest from repo root:

```bash
python3 -m pytest src/pingpong_moz1/test -q
python3 -m pytest src/pingpong_moz1/test/test_policy_profile_contract.py -q
python3 -m pytest src/pingpong_moz1/test/test_policy_profile_contract.py::ObservationProfileContractTest::test_ideal_profile_uses_causal_racket_velocity_rel_z_and_age -q
python3 -m pytest src/pingpong_replay/test -q
python3 -m pytest src/pingpong_hil_rl/test -q
```

Colcon tests:

```bash
colcon test --packages-select pingpong_controller --event-handlers console_direct+
colcon test-result --verbose
```

Some tests are fixture-gated and skip unless simcode checkpoint/model artifacts are present. Prefer preserving these skip guards rather than weakening checkpoint contracts.

## RMP validation and simulation commands

Validate recovered RMP against Movax DataTracer output when a matching `.dat` file is available:

```bash
python3 -m recovered_rmp.extract_movax_datatracer <dat-file> --output /tmp/right_arm_trace
python3 -m recovered_rmp.validate_rmp_datatracer \
  /tmp/right_arm_trace.npz <probe_recording_dir> \
  movax/controller/configuration/model/user/spirit01_base_e_no_neck_hh.yaml \
  --include-feedback-reference-baseline
```

Replay a real recording through recovered RMP and MuJoCo ideal PD:

```bash
python simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/replay_rmp_recording.py \
  <control_samples.jsonl-or-recording-dir> \
  --calibration-profile recovered_yaml_pd_fit \
  --output /tmp/rmp_replay
```

Replay through the full MJX environment path:

```bash
cd simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim
python replay_rmp_recording_mjx.py <control_samples.jsonl> \
  --calibration-profile recovered_yaml_pd_fit \
  --output /tmp/rmp_mjx_replay
```

Run RMP/PD calibration scans:

```bash
cd simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim
python calibrate_rmp_pd.py --output /tmp/rmp_pd_cpu.json
JAX_PLATFORMS=cpu python calibrate_rmp_pd_mjx.py --output /tmp/rmp_pd_mjx.json
```

Common overrides for deeper scans include `--rmp-jnt-kp-scale`, `--rmp-jnt-kd-scale`, `--controller-kp-scale`, `--controller-kd-scale`, and `--acceleration-feedforward-s2`.

When tuning XML PD parameters for RMP equivalence, create a new XML file (for example `moz1_rmp_equiv_experiment.xml`) instead of modifying the existing deployment XMLs used by current scripts/checkpoints.

The current MJX-equivalence workflow is:

1. Run the tick-alignment helper in `src/pingpong_moz1/pingpong_controller/tools/rl_sim/analyze_rmp_tick_alignment.py` to inspect `command.acceleration`, `publishedSafePosition`, and `ros.jointAcceleration` around divergence onset.
2. Run the MJX scan wrapper in `src/pingpong_moz1/pingpong_controller/tools/rl_sim/scan_rmp_equiv_mjx.py`, which always targets `moz1_rmp_equiv.xml` and reuses the existing simcode5 calibration logic.
3. Compare q / dq / qdd together; do not optimize only q while ignoring acceleration.

## Runtime / experiment entrypoints

Interactive policy safety probe and analysis:

```bash
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
ANALYSIS=latest ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
PROBE_CONFIG_ONLY=true ./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

Policy analysis and safety scenario scripts:

```bash
./RUN_POLICY_ANALYSIS.sh
./RUN_POLICY_SAFETY_SCENARIOS.sh
./RUN_POLICY_SAFETY_SCENARIOS.sh up_occlusion
./RUN_REAL_POLICY_EXPERIMENT.sh
```

Camera/detector and robot execution examples are documented in [readme.md](readme.md). The important operational pattern is: source ROS, build changed packages, then run `ros2 run pingpong_controller ...` or `ros2 run pingpong_replay ...` with explicit parameters.

## High-level architecture

- [src/pingpong_moz1/](src/pingpong_moz1/) is the main ROS package (`pingpong_controller`). Its main entrypoint is `pingpong_node`, with supporting modules for ball detection, calibration loading, RL policy wrappers, state estimation, safety limiting, mocap input, replay/probe tools, and MJX policy deployment.
- [src/pingpong_moz1/pingpong_controller/tools/interactive_policy_safety_probe.py](src/pingpong_moz1/pingpong_controller/tools/interactive_policy_safety_probe.py) is the large interactive web/probe harness used for dry-runs, camera/mocap ball input, ROS command publishing, sim mirror, racket tracking, MPC juggling, safety modes, and recording. Many shell scripts configure this one tool via environment variables.
- [src/pingpong_moz1/pingpong_controller/ball_tracker/](src/pingpong_moz1/pingpong_controller/ball_tracker/) contains the higher-level ball tracking stack: dynamics model, EKF tracker, impact prediction, config, logging, and RL observation helpers. Keep detection-frame and control-frame semantics explicit; live and offline record→annotate paths are expected to match.
- [src/pingpong_moz1/pingpong_controller/tools/rl_sim/](src/pingpong_moz1/pingpong_controller/tools/rl_sim/) is the in-repo MuJoCo/MJX RL simulation area. [mjx_juggle_env.py](src/pingpong_moz1/pingpong_controller/tools/rl_sim/mjx_juggle_env.py) is the main environment implementation; training/validation scripts live beside it.
- [simcode2/](simcode2/), [simcode3/](simcode3/), [simcode4/](simcode4/), [simcode5/](simcode5/), and [simcode6/](simcode6/) are historical or parallel simulation/deployment snapshots. Tests in [src/pingpong_moz1/test/test_policy_profile_contract.py](src/pingpong_moz1/test/test_policy_profile_contract.py) encode checkpoint, XML, observation, and actuator/servo profile contracts across these snapshots.
- [recovered_rmp/](recovered_rmp/) is the recovered Movax RMP motion-mode implementation. It is pure Python/NumPy plus optional JAX and MuJoCo adapters, not a ROS package.
- [src/pingpong_replay/](src/pingpong_replay/) provides ROS replay of recorded, compensated, or sim-actual trajectories and plots replay-vs-real analysis outputs.
- [src/pingpong_hil_rl/](src/pingpong_hil_rl/) contains HIL RL package code and tests.
- [analysis_results/](analysis_results/) and package `outputs/` directories contain experiment artifacts. Avoid treating generated plots, caches, pyc files, and recordings as source code.

## Recovered RMP implementation notes

The recovered RMP algorithm is documented in detail in [recovered_rmp/README.md](recovered_rmp/README.md), [recovered_rmp/RECOVERY_EVIDENCE.md](recovered_rmp/RECOVERY_EVIDENCE.md), and [recovered_rmp/CALIBRATION_RESULTS.md](recovered_rmp/CALIBRATION_RESULTS.md). Important implementation structure:

- [recovered_rmp/core.py](recovered_rmp/core.py) implements the acceleration-level RMP/QP equations: task and joint reference acceleration, predictive joint acceleration bounds, QP construction, and the small box-QP active-set solver.
- [recovered_rmp/trajectory.py](recovered_rmp/trajectory.py) reconstructs the stateful 1 kHz joint RMP motion mode (`JointRMPPlanner`): target rate limiting, event-driven constant-acceleration Kalman target estimation, target/output ring buffers, QP solve, and integration of q/qd/qdd.
- [recovered_rmp/jax_backend.py](recovered_rmp/jax_backend.py) mirrors the joint RMP path in JAX for MJX/batched training. Keep NumPy/JAX parity tests passing when changing RMP behavior.
- [recovered_rmp/mujoco_adapter.py](recovered_rmp/mujoco_adapter.py) replaces private `rkdl` with MuJoCo FK/Jacobian/site kinematics.
- [recovered_rmp/replay.py](recovered_rmp/replay.py) loads real `control_samples.jsonl`, aligns acceleration using forward velocity differences on contiguous `sourceSeq`, and replays `recorded q target -> recovered RMP q/qd/qdd -> ideal torque PD -> MuJoCo`.
- [recovered_rmp/validate_rmp_datatracer.py](recovered_rmp/validate_rmp_datatracer.py) is the key tool for separating recovered RMP output error from lower-level servo/encoder feedback error.

Current known calibration state from [recovered_rmp/CALIBRATION_RESULTS.md](recovered_rmp/CALIBRATION_RESULTS.md): the recovered RMP output layer has matched DataTracer `axis_pos/axis_vel/axis_acc` within the stated thresholds on one stable segment, but full encoder feedback equivalence is not yet 1:1. Do not claim the sim has no sim2real gap unless the q/dq/ddq and racket end-effector metrics pass on held-out real recordings.

## Sim/real metric conventions

- Use SI units internally: radians, rad/s, rad/s², meters, seconds.
- For acceleration metrics, use the repository’s forward-difference convention: real velocity differentiated by `ros.jointSampleStampS`, simulation velocity by `sourceTimeS`, and only contiguous `sourceSeq` samples are valid. Do not substitute smoothed central differences when comparing against existing reports.
- Keep comparison layers separate: RMP output vs DataTracer axis reference, MuJoCo/MJX simulated joint feedback vs robot encoder feedback, and racket `right_ee_site` kinematics are different metrics.
- For racket kinematics, the racket center used by control/FK is `right_ee_site`, not the `right_racket` body origin.

## Safety and deployment constraints

This code can publish commands to real robot hardware. Before changing deployment behavior, identify whether the path affects dry-run only, sim mirror only, or real ROS publishing. Preserve fail-closed behavior for stale ball state, stale joint state, unsupported frames, hard workspace stops, command lease timeout, and operator-controlled `Send command` / start buttons.

`ROS_PUBLISH_COMMANDS=true` only gives the web/probe permission to publish; the UI `Send command` and mode-specific start controls are still operator gates and should remain off by default.

## Documentation sources to consult

- [readme.md](readme.md): camera calibration, detector launch, depth assist, mocap input, replay, real experiment scripts, interactive probe settings, racket tracking, and MPC juggling notes.
- [recovered_rmp/README.md](recovered_rmp/README.md): recovered RMP equations, runtime filters, simcode5/MJX integration, and validation commands.
- [recovered_rmp/CALIBRATION_RESULTS.md](recovered_rmp/CALIBRATION_RESULTS.md): current RMP/MJX calibration status, metrics, and missing data needed for true 1:1 hardware equivalence.
