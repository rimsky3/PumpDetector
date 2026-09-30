# Experiment: two-tier signal — 15m candidate, hourly confirmation

14 tickers, 2026-08-07 – 2026-09-30, 49 labeled pumps (impulse labels). Out-of-fold LightGBM scores (leave-ticker-out, nested threshold), normalised by threshold. Each 15m bar gets the hourly score of the last hourly bar closed by the 15m bar's close, so no look-ahead.

- **15m (candidate)**: 15m model alone (liquidations + short features)
- **15m & 1h ≥ c**: 15m signal and the hourly score is at least c × its threshold
- **… within 4 h**: the hourly condition held at least once in the last 4 closed hours
- **1h only**: hourly model alone, on the 15m grid

|  | pumps_caught | false_starts_triggered | median_delay_min | median_move_missed | median_upside_left | false_alarms_per_30d |
|---|---|---|---|---|---|---|
| 15m (candidate) | 47/49 | 12/19 | 30.000 | 0.220 | 0.243 | 5.633 |
| 15m & 1h ≥ 0.5 | 47/49 | 8/19 | 45.000 | 0.262 | 0.213 | 4.460 |
| 15m & 1h ≥ 0.75 | 47/49 | 8/19 | 50.000 | 0.264 | 0.213 | 3.756 |
| 15m & 1h ≥ 1 | 44/49 | 7/19 | 60.000 | 0.322 | 0.193 | 2.958 |
| 15m & 1h ≥ 0.5 within 4 h | 47/49 | 8/19 | 45.000 | 0.262 | 0.242 | 4.694 |
| 15m & 1h ≥ 1 within 4 h | 44/49 | 7/19 | 55.000 | 0.322 | 0.193 | 2.817 |
| 1h only | 44/49 | 8/19 | 45.000 | 0.288 | 0.211 | 4.272 |

Median wait from Candidate to Alert (1h ≥ 0.5 within 4 h) on pumps caught by both: 0 min (47 pumps).
