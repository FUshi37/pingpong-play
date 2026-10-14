# Three-day offline ball trajectory study

- Dates: 20260810, 20260811, 20260812
- Runs: 66 eligible / 68 discovered
- Effective juggling duration: 151.447735 s
- Fresh camera samples: 13574
- Numeric outliers replaced in the acausal reference: 178
- Full raw IR/depth rerun available: 0/68 runs

## Aggregate detector-state replay

| chain | position p95 median across runs (mm) | velocity p95 median across runs (m/s) | position HF p95 (mm) | velocity HF p95 (m/s) | position jumps | velocity jumps |
|---|---:|---:|---:|---:|---:|---:|
| recorded policy input | 12.738259 | 0.558413 | 41.967744 | 0.316852 | 61.0 | 33.0 |
| legacy_endpoint | 17.492033 | 0.451045 | 25.511713 | 0.350915 | 11.0 | 36.0 |
| impact_endpoint | 16.651863 | 0.394432 | 24.646673 | 0.313472 | 7.0 | 16.0 |
| physics_only | 15.774485 | 0.433117 | 24.006206 | 0.31828 | 6.0 | 0.0 |
| physics_gate | 15.693922 | 0.405138 | 23.517016 | 0.308084 | 0.0 | 0.0 |
| current_latest | 15.87682 | 0.405138 | 22.917874 | 0.308084 | 0.0 | 0.0 |
| position_smooth_20mm | 15.87682 | 0.405138 | 22.952264 | 0.308084 | 0.0 | 0.0 |
| velocity_gain_055 | 15.87682 | 0.412278 | 22.917874 | 0.31828 | 0.0 | 0.0 |
| recommended | 15.87682 | 0.405138 | 22.917874 | 0.308084 | 0.0 | 0.0 |

`recommended` keeps synchronized IR+depth, least-squares velocity, reacquisition velocity handover and control-time compensation off. It enables loss-boundary clearing, the ballistic motion gate, causal physics velocity, bounded endpoint position smoothing and confirmed impact reset.

The clean trajectory is acausal and is used only as an evaluation reference/counterfactual replay input. It is not deployable online.
