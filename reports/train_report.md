# Training report

Generated 2026-10-01 10:58 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 19309 hourly bars, 15 tickers (AGT, AIN, AKE, ARK, G, LSK, MOVR, MUBARAK, NIGHT, NMR, NOM, PHA, QNT, SOON, TRUMP), 361 pump hours (1.9%), 62 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.270 | 0.490 | 0.348 | — | 41/62 | 0.000 | 5/18 | 3.207 |
| logreg | 0.384 | 0.507 | 0.437 | 0.366 | 49/62 | 0.000 | 8/18 | 3.692 |
| lightgbm | 0.366 | 0.535 | 0.434 | 0.411 | 48/62 | 0.000 | 8/18 | 4.437 |

## Training settings

- `--fresh-hours 48`: 630 unlabeled hours after 2026-09-29 10:57 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.182 | 1/2 | 0.000 | 0/0 | 1.731 |
| AIN | 0.400 | 4/4 | 0.000 | 0/0 | 1.121 |
| AKE | 0.495 | 5/10 | 0.750 | 1/3 | 1.579 |
| ARK | 0.224 | 4/6 | 2.750 | 0/2 | 4.601 |
| G | 0.349 | 3/7 | 0.583 | 0/1 | 2.105 |
| LSK | 0.119 | 1/2 | 0.000 | 2/2 | 3.158 |
| MOVR | 0.429 | 1/1 | 0.500 | 0/0 | 7.411 |
| MUBARAK | 0.414 | 9/10 | 0.000 | 2/5 | 4.489 |
| NIGHT | 0.300 | 1/1 | 0.750 | 0/0 | 6.231 |
| NMR | 0.200 | 2/2 | 0.000 | 0/0 | 3.928 |
| NOM | 0.091 | 2/7 | 0.000 | 0/1 | 1.155 |
| PHA | 0.214 | 2/2 | 5.792 | 0/1 | 5.201 |
| QNT | 0.493 | 2/3 | 2.917 | 0/1 | 3.925 |
| SOON | 0.606 | 2/3 | 1.500 | 0/2 | 1.130 |
| TRUMP | 0.526 | 2/2 | 1.250 | 0/0 | 0.567 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.444 | 2/2 | 0.000 | 0/0 | 4.038 |
| AIN | 0.182 | 2/4 | 2.167 | 0/0 | 2.804 |
| AKE | 0.483 | 9/10 | 0.000 | 2/3 | 4.737 |
| ARK | 0.319 | 5/6 | 1.750 | 0/2 | 5.176 |
| G | 0.273 | 3/7 | 0.583 | 0/1 | 1.053 |
| LSK | 0.459 | 2/2 | 0.000 | 2/2 | 6.842 |
| MOVR | 0.545 | 1/1 | 0.000 | 0/0 | 5.701 |
| MUBARAK | 0.517 | 10/10 | 0.000 | 3/5 | 2.806 |
| NIGHT | 0.244 | 1/1 | 0.000 | 0/0 | 5.098 |
| NMR | 0.258 | 2/2 | 0.042 | 0/0 | 4.489 |
| NOM | 0.200 | 3/7 | 0.000 | 0/1 | 2.887 |
| PHA | 0.652 | 2/2 | 0.042 | 0/1 | 3.467 |
| QNT | 0.556 | 3/3 | 0.833 | 0/1 | 3.925 |
| SOON | 0.407 | 2/3 | 2.500 | 1/2 | 2.261 |
| TRUMP | 0.750 | 2/2 | 2.250 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.222 | 1/2 | 0.750 | 0/0 | 1.154 |
| AIN | 0.286 | 4/4 | 0.292 | 0/0 | 0.561 |
| AKE | 0.413 | 6/10 | 0.750 | 2/3 | 4.211 |
| ARK | 0.491 | 6/6 | 1.250 | 0/2 | 5.751 |
| G | 0.545 | 6/7 | 1.083 | 0/1 | 2.632 |
| LSK | 0.452 | 2/2 | 0.000 | 2/2 | 4.737 |
| MOVR | 0.561 | 1/1 | 0.000 | 0/0 | 8.551 |
| MUBARAK | 0.451 | 10/10 | 0.000 | 3/5 | 9.540 |
| NIGHT | 0.209 | 1/1 | 0.000 | 0/0 | 13.029 |
| NMR | 0.286 | 2/2 | 0.042 | 0/0 | 3.367 |
| NOM | 0.182 | 2/7 | 0.000 | 0/1 | 1.732 |
| PHA | 0.656 | 2/2 | 0.000 | 0/1 | 8.090 |
| QNT | 0.603 | 3/3 | 0.833 | 0/1 | 2.243 |
| SOON | 0.367 | 2/3 | 1.625 | 1/2 | 1.130 |
| TRUMP | 0.000 | 0/2 | — | 0/0 | 0.000 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| oi_chg_24h | 0.357 | 34.131 |
| volume_z_7d | 1.613 | 24.172 |
| ret_24h | 0.577 | 7.672 |
| oi_z_24h | 0.353 | 5.240 |
| oi_chg_4h | 0.091 | 3.742 |
| ret_4h | 0.113 | 3.384 |
| oi_chg_1h | 0.225 | 3.127 |
| ls_long_pct | -0.640 | 2.896 |
| buy_ratio_4h | 0.300 | 2.145 |
| liq_short_4h_oi | -0.282 | 2.125 |
| funding_z_7d | 0.023 | 1.619 |
| volatility_24h | -0.441 | 1.407 |
| liq_long_4h_oi | -0.559 | 1.308 |
| oi_minus_ret_24h | -0.040 | 1.233 |
| ret_1h | 0.170 | 1.182 |
| oi_z_7d | 0.113 | 1.122 |
| volume_ratio_24h | 0.088 | 0.809 |
| ls_chg_24h | 0.391 | 0.637 |
| funding | -0.138 | 0.578 |
| range_1h | -0.211 | 0.445 |
| liq_imbalance_4h | 0.199 | 0.419 |
| funding_chg_24h | -0.134 | 0.305 |
| buy_ratio_1h | 0.114 | 0.302 |

## 15m model and signal tiers

LightGBM on 66837 15-minute bars (34 features: the hourly set with the same wall-clock windows plus short bar features), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 1.94 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.395 | 0.563 | 0.464 | 0.398 | 58/62 | 0.292 | 12/18 | 7.196 |
| 15m + 1h ≥ 0.75 (alert) | 0.415 | 0.516 | 0.460 | — | 55/62 | 0.583 | 7/18 | 5.128 |
