# Three-day offline ball trajectory study

- Dates: 20260810
- Runs: 15 eligible / 16 discovered
- Effective juggling duration: 30.429938 s
- Fresh camera samples: 2737
- Numeric outliers replaced in the acausal reference: 22
- Full raw IR/depth rerun available: 0/16 runs

## Aggregate detector-state replay

| chain | position p95 median across runs (mm) | velocity p95 median across runs (m/s) | position HF p95 (mm) | velocity HF p95 (m/s) | position jumps | velocity jumps |
|---|---:|---:|---:|---:|---:|---:|
| recorded policy input | 10.901198 | 0.360638 | 34.706536 | 0.303408 | 12.0 | 7.0 |
| legacy_endpoint | 16.08873 | 0.315477 | 23.272351 | 0.316595 | 2.0 | 8.0 |
| impact_endpoint | 15.40411 | 0.274192 | 23.012216 | 0.321987 | 0.0 | 3.0 |
| physics_only | 15.731929 | 0.387246 | 23.492091 | 0.314568 | 0.0 | 0.0 |
| physics_gate | 15.731929 | 0.387246 | 21.713446 | 0.402891 | 0.0 | 0.0 |
| current_latest | 15.960273 | 0.387246 | 20.731952 | 0.402891 | 0.0 | 0.0 |
| position_smooth_20mm | 15.960273 | 0.387246 | 20.731952 | 0.402891 | 0.0 | 0.0 |
| velocity_gain_055 | 15.960273 | 0.387246 | 20.731952 | 0.402891 | 0.0 | 0.0 |
| recommended | 15.960273 | 0.387246 | 20.731952 | 0.402891 | 0.0 | 0.0 |

`recommended` keeps synchronized IR+depth, least-squares velocity, reacquisition velocity handover and control-time compensation off. It enables loss-boundary clearing, the ballistic motion gate, causal physics velocity, bounded endpoint position smoothing and confirmed impact reset.

The clean trajectory is acausal and is used only as an evaluation reference/counterfactual replay input. It is not deployable online.
