# Three-day offline ball trajectory study

- Dates: 20260811
- Runs: 25 eligible / 26 discovered
- Effective juggling duration: 63.020305 s
- Fresh camera samples: 5675
- Numeric outliers replaced in the acausal reference: 84
- Full raw IR/depth rerun available: 0/26 runs

## Aggregate detector-state replay

| chain | position p95 median across runs (mm) | velocity p95 median across runs (m/s) | position HF p95 (mm) | velocity HF p95 (m/s) | position jumps | velocity jumps |
|---|---:|---:|---:|---:|---:|---:|
| recorded policy input | 13.595851 | 0.61655 | 42.670571 | 0.370106 | 28.0 | 20.0 |
| legacy_endpoint | 18.798255 | 0.446479 | 24.005344 | 0.361903 | 8.0 | 19.0 |
| impact_endpoint | 17.296032 | 0.400409 | 23.80407 | 0.352878 | 6.0 | 12.0 |
| physics_only | 15.558554 | 0.511163 | 22.464922 | 0.389394 | 6.0 | 0.0 |
| physics_gate | 15.558554 | 0.403165 | 22.243716 | 0.301599 | 0.0 | 0.0 |
| current_latest | 15.793367 | 0.403165 | 22.046249 | 0.301599 | 0.0 | 0.0 |
| position_smooth_20mm | 15.793367 | 0.403165 | 22.046249 | 0.301599 | 0.0 | 0.0 |
| velocity_gain_055 | 15.793367 | 0.415532 | 22.046249 | 0.301599 | 0.0 | 0.0 |
| recommended | 15.793367 | 0.403165 | 22.046249 | 0.301599 | 0.0 | 0.0 |

`recommended` keeps synchronized IR+depth, least-squares velocity, reacquisition velocity handover and control-time compensation off. It enables loss-boundary clearing, the ballistic motion gate, causal physics velocity, bounded endpoint position smoothing and confirmed impact reset.

The clean trajectory is acausal and is used only as an evaluation reference/counterfactual replay input. It is not deployable online.
