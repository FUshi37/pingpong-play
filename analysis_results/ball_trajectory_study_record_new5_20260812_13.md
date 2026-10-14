# Three-day offline ball trajectory study

- Dates: 20260812, 20260813
- Runs: 23 eligible / 23 discovered
- Effective juggling duration: 54.245025 s
- Fresh camera samples: 4262
- Numeric outliers replaced in the acausal reference: 49
- Full raw IR/depth rerun available: 0/23 runs

## Aggregate detector-state replay

| chain | position p95 median across runs (mm) | velocity p95 median across runs (m/s) | position HF p95 (mm) | velocity HF p95 (m/s) | position jumps | velocity jumps |
|---|---:|---:|---:|---:|---:|---:|
| recorded policy input | 12.860851 | 0.88913 | 47.769521 | 0.307022 | 8.0 | 2.0 |
| legacy_endpoint | 21.198361 | 0.835823 | 30.402332 | 0.590603 | 0.0 | 6.0 |
| impact_endpoint | 18.790318 | 0.58794 | 30.384147 | 0.337329 | 0.0 | 0.0 |
| physics_only | 18.343024 | 0.561308 | 30.197648 | 0.352688 | 0.0 | 0.0 |
| physics_gate | 18.211899 | 0.474736 | 30.197648 | 0.301231 | 0.0 | 0.0 |
| current_latest | 18.067937 | 0.474736 | 32.883369 | 0.301231 | 0.0 | 0.0 |
| position_smooth_20mm | 18.067937 | 0.474736 | 32.145844 | 0.301231 | 0.0 | 0.0 |
| velocity_gain_055 | 18.067937 | 0.497434 | 32.883369 | 0.298622 | 0.0 | 0.0 |
| recommended | 18.067937 | 0.474736 | 32.883369 | 0.301231 | 0.0 | 0.0 |

`recommended` keeps synchronized IR+depth, least-squares velocity, reacquisition velocity handover and control-time compensation off. It enables loss-boundary clearing, the ballistic motion gate, causal physics velocity, bounded endpoint position smoothing and confirmed impact reset.

The clean trajectory is acausal and is used only as an evaluation reference/counterfactual replay input. It is not deployable online.
