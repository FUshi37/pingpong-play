# Raw vs cleaned real-trajectory policy replay

- Dates: 20260810, 20260811, 20260812
- Runs: 66 eligible / 68 discovered
- Effective juggling duration: 151.447735 s
- Policy ticks per branch: 29565
- Camera numeric outliers robustly replaced: 178
- Policy ticks with anomalous/missing raw input: 2248

## Paired policy response

| metric | raw vs clean |
|---|---:|
| action difference RMS | 0.721944 |
| action difference p95 | 1.483783 |
| command difference RMS (rad) | 0.200797 |
| racket difference RMS (mm) | 36.608345 |
| racket difference p95 (mm) | 78.960036 |
| racket difference max (mm) | 270.729532 |
| action difference p95, 0-100 ms after anomaly | 1.804298 |
| action difference p95, baseline | 1.181547 |
| racket difference p95, 0-100 ms after anomaly (mm) | 113.024161 |
| racket difference p95, baseline (mm) | 65.094223 |

## Smoothness and catch geometry

| metric | raw | clean |
|---|---:|---:|
| action-rate p95 | 93.829195 | 76.243233 |
| command-rate p95 (rad/s) | 4.741079 | 4.801799 |
| best descending XY error p50 (mm) | 16.485175 | 14.781582 |
| best descending XY error p95 (mm) | 89.180869 | 98.70622 |
| descents within 100 mm | 297/321 | 287/321 |

The ball is forced and catch geometry is evaluated against the same clean reference in both branches. Counts are sensitivity indicators, not predicted real-world success rates.
