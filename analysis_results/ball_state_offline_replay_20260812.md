# Offline ball-state replay

- Runs: 41
- Fresh camera samples: 54884
- Full IR/depth rerun available: 0/41 runs

## Clean recovery model comparison

| group | events | recorded median norm (mm) | revised median norm (mm) | paper quadratic median norm (mm) |
|---|---:|---:|---:|---:|
| high-FOV | 33 | 26.66 | 19.276 | 29.549 |
| depth fault | 25 | 39.557 | 40.605 | 39.815 |

## State-transition audit

- Recorded stale warm-up bridges rejected by the revised 120 mm gate: 19
- First impact candidates previously starved by motion gating: 9
- Pending impact confirmations previously starved by motion gating: 1
- Revised predictor non-finite outputs: 0
- Revised predictor large prediction-to-sensor handovers (>50 mm position innovation or >0.75 m/s velocity innovation): 6/67
  - record_new4 (historical detector velocity retained): 6
  - 20260812 recordings: 0

The state replay uses saved detector telemetry. Annotated JPEGs do not contain the original 16-bit IR/depth arrays, so runs without `.npy` or `.bag` files cannot rerun contour extraction or stereo sampling.
