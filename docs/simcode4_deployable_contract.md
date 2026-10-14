# simcode4 frozen and deployable plant contracts

## Frozen checkpoint contract

The submitted `mjx_curriculum_best.pkl` was trained and validated with this
causal plant order:

```text
policy integration
-> actual-feedback inverse MPC
-> command delay / FOPDT
-> target-aware servo planner (velocity 1.0, acceleration 0.8)
-> XML position PD
```

This remains the default in `simcode4` and is the authoritative mirror for the
submitted checkpoint. `RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` therefore forces
the simcode4 mirror to use the original MJX environment rather than the
handwritten CPU mirror.

## Real hardware default contract

`MCControlNode` and the physical joint interface accept processed position
targets; they do not expose a planner after the physical actuator dynamics.
The probe therefore defaults to no software servo planner:

```text
policy integration
-> actual-feedback inverse MPC
-> published joint angle
-> physical joint delay / lag
```

The hardware process must not replay the 72 ms delay or FOPDT in software. The
interactive probe reports the real planner as disabled and publishes the
compensated angle for the frozen V5 checkpoint. It still keeps V5's original
FOPDT-then-planner order inside the authoritative MJX mirror.

Setting `SIMCODE4_REAL_SERVO_PLANNER=true` explicitly enables this alternative
real order:

```text
policy integration
-> actual-feedback inverse MPC
-> target-aware servo planner (velocity 1.0, acceleration 0.8)
-> published joint angle
-> physical joint delay / lag
```

That opt-in remains fail-closed unless the selected checkpoint contains
`arm_servo_target_planner_before_actuator=true`, because the frozen V5 policy
was not trained with the planner in that location.

## Retraining profile

The MJX environment has an opt-in
`arm_servo_target_planner_before_actuator` field. Its default is `false`, so old
checkpoints and frozen validation remain unchanged. The curriculum profile

```text
goal_d455_autolaunch_viewdense_constrained_mpc_launch17_obsres2mm_servo_deployable_v6
```

inherits the complete measured-observation V5 curriculum and changes only the
planner/actuator order. It is intended for continuation training from the
submitted V5 checkpoint. The old checkpoint is not directly valid under this
new plant: a deterministic seed-20260729 A/B terminated at step 69 with one hit
after changing only the order, demonstrating that continuation training is
required rather than optional tuning.

`RUN_SIMCODE4_DEPLOYABLE_V6_CONTINUATION.sh` launches full continuation with
the submitted run's PPO hyperparameters. `SIMCODE4_RESUME_FROM`,
`SIMCODE4_OUTPUT_DIR`, `SIMCODE4_GPU`, and `SIMCODE4_WANDB` may be overridden.
The one-update smoke checkpoint is diagnostic only: its deterministic
seed-20260729 evaluation still terminates at step 69 with one hit.

For evaluation without changing a checkpoint, pass
`--arm-servo-target-planner-before-actuator` to
`validate_juggle_mjx_ppo.py`. Omitting it reproduces the frozen plant.

## 67D deployment observation fixes

- `obs[14:17]` and `obs[17:20]` use the XML 700-step warm-start base state,
  rather than hard-coded zeros.
- `obs[50]` reproduces MJX float32 `tau_ms * 1e-3 / delay_max_s` arithmetic.
- simcode4 racket velocity uses the fixed 5 ms policy tick, matching training.
- joint position is radians, velocity is radians/second, Cartesian quantities
  are base-yaw-frame metres and metres/second, and ball age is held at zero for
  non-missing 60 Hz samples.

The frozen trace regression covers `obs[0:65]`; phase fields `obs[65:67]` use
episode-randomized gravity in simulation and physical gravity on hardware.
