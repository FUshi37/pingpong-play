# Ball State Estimator

This package is the backend-neutral estimator layer between low-latency IR
position measurements and the control policy. The default backend is a
nine-state hybrid EKF with state `[px, py, pz, vx, vy, vz, ax, ay, az]`.

All estimator inputs and outputs use SI units. Convert the D455/IR detector's
millimetres to metres at the bridge boundary.

```python
from pingpong_controller.ball_tracker import BallStateEstimator

estimator = BallStateEstimator()

# Camera callback: use the image timestamp, not detector-completion time.
estimator.update(
    ir_position_mm * 1e-3,
    image_timestamp_s,
    racket_pose={"position": racket_position_m, "normal": [0, 0, 1]},
)

# 200 Hz control callback. Prediction is non-mutating, so a delayed image
# timestamp can still be fused correctly when that camera callback arrives.
estimate = estimator.predict_to(
    control_timestamp_s,
    racket_pose={"position": racket_position_m, "normal": [0, 0, 1]},
)
if estimate.valid:
    policy_position = estimate.position
    policy_velocity = estimate.velocity
    impact = estimate.impact_prediction
```

The policy observation adapter preserves current filtered position and
velocity, then appends time-to-contact and impact position. Enabling the
optional acceleration/uncertainty fields requires training a checkpoint with
that observation contract. Existing 50/67-dimensional checkpoints must not be
silently fed the enlarged vector.

The EKF backend handles flight, near-racket contact candidates, and a hard
post-impact velocity reset inferred from successive IR measurements. During a
measurement dropout it predicts for up to 200 ms by default, with covariance
growth visible through `position_std` and `velocity_std`; it then invalidates
and resets the track.

The interactive safety probe keeps both the estimator master switch and this
backend selector off by default. Enable the complete hybrid prediction path
explicitly for a manual experiment:

```bash
ENABLE_BALL_STATE_ESTIMATOR=true \
ENABLE_HYBRID_BALL_TRACKER=true \
BALL_ESTIMATOR_ENABLE_PREDICTION=true \
./RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh
```

Set `HYBRID_BALL_TRACKER_CONFIG_PATH=/path/to/config.yaml` to use a different
tuning file. Enabling `ENABLE_BALL_STATE_ESTIMATOR` alone continues to select
the legacy estimator, so existing experiments remain reproducible.

The probe's right-side **Node Flow** panel also exposes a **hybrid EKF**
checkbox. It is disabled until **3D filter** is enabled. Changing the checkbox
switches backend on the next control tick and intentionally starts a fresh
ball track; **predict / bridge** independently controls control-time and racket
contact prediction. The browser checkbox inherits the launch defaults above,
so it is unchecked when the script is run without environment overrides.

Validation commands:

```bash
PYTHONPATH=src/pingpong_moz1 python3 \
  src/pingpong_moz1/pingpong_controller/tools/ball_tracker_simulation.py

PYTHONPATH=src/pingpong_moz1 python3 \
  src/pingpong_moz1/pingpong_controller/tools/ball_tracker_robot_example.py
```

`BallExperimentLogger` records raw measurements, ground truth, all nine state
components, uncertainties, impact predictions, and errors. For the real/sim
RL comparison, `RLPerformanceLogger` writes one row per episode for the
`raw`, `ekf`, and `ekf_impact` variants and summarizes hits, maximum continuous
juggling time, impact error, and failure cases.
