# Training report

Generated 2026-10-02 19:27 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 21050 hourly bars, 16 tickers (AGT, AIN, AKE, ARK, G, LSK, MOVR, MUBARAK, NIGHT, NMR, NOM, PHA, QNT, SOON, TRUMP, ZEC), 435 pump hours (2.1%), 72 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.294 | 0.492 | 0.368 | — | 48/71 | 0.083 | 5/22 | 3.044 |
| logreg | 0.356 | 0.501 | 0.416 | 0.330 | 58/71 | 0.042 | 9/22 | 4.070 |
| lightgbm | 0.366 | 0.529 | 0.433 | 0.387 | 58/71 | 0.042 | 12/22 | 5.131 |

## Training settings

- `--fresh-hours 48`: 684 unlabeled hours after 2026-09-30 19:26 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.143 | 1/2 | 0.000 | 0/0 | 2.252 |
| AIN | 0.400 | 4/4 | 0.000 | 0/0 | 1.093 |
| AKE | 0.495 | 5/10 | 0.750 | 1/3 | 1.542 |
| ARK | 0.222 | 4/6 | 2.750 | 0/2 | 5.062 |
| G | 0.349 | 3/7 | 0.583 | 0/1 | 2.056 |
| LSK | 0.414 | 2/2 | 0.000 | 2/2 | 2.570 |
| MOVR | 0.375 | 1/1 | 0.500 | 0/0 | 7.312 |
| MUBARAK | 0.414 | 9/10 | 0.000 | 2/5 | 4.377 |
| NIGHT | 0.269 | 1/1 | 0.750 | 0/0 | 6.192 |
| NMR | 0.200 | 2/2 | 0.000 | 0/0 | 3.830 |
| NOM | 0.091 | 2/7 | 0.000 | 0/1 | 1.130 |
| PHA | 0.214 | 2/2 | 5.792 | 0/1 | 5.094 |
| QNT | 0.493 | 2/3 | 2.917 | 0/1 | 3.827 |
| SOON | 0.454 | 3/5 | 0.000 | 0/2 | 1.102 |
| TRUMP | 0.526 | 2/2 | 1.250 | 0/0 | 0.553 |
| ZEC | 0.523 | 5/7 | 0.500 | 0/4 | 1.102 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.286 | 2/2 | 0.000 | 0/0 | 3.941 |
| AIN | 0.111 | 2/4 | 2.167 | 0/0 | 2.733 |
| AKE | 0.385 | 9/10 | 0.000 | 2/3 | 9.764 |
| ARK | 0.280 | 4/6 | 2.250 | 0/2 | 5.625 |
| G | 0.431 | 5/7 | 1.583 | 0/1 | 0.514 |
| LSK | 0.500 | 2/2 | 0.000 | 2/2 | 5.653 |
| MOVR | 0.545 | 1/1 | 0.000 | 0/0 | 5.062 |
| MUBARAK | 0.479 | 10/10 | 0.000 | 2/5 | 2.736 |
| NIGHT | 0.270 | 1/1 | 0.000 | 0/0 | 2.815 |
| NMR | 0.308 | 2/2 | 0.042 | 0/0 | 3.830 |
| NOM | 0.188 | 3/7 | 0.000 | 0/1 | 3.391 |
| PHA | 0.640 | 2/2 | 0.042 | 0/1 | 5.094 |
| QNT | 0.610 | 3/3 | 0.833 | 0/1 | 4.374 |
| SOON | 0.282 | 3/5 | 3.000 | 1/2 | 1.653 |
| TRUMP | 0.750 | 2/2 | 2.250 | 0/0 | 0.000 |
| ZEC | 0.516 | 7/7 | 0.500 | 2/4 | 7.712 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.143 | 1/2 | 0.750 | 0/0 | 1.126 |
| AIN | 0.303 | 4/4 | 0.292 | 0/0 | 0.000 |
| AKE | 0.455 | 7/10 | 0.167 | 2/3 | 3.084 |
| ARK | 0.435 | 5/6 | 1.250 | 0/2 | 5.062 |
| G | 0.545 | 6/7 | 1.083 | 0/1 | 2.570 |
| LSK | 0.418 | 2/2 | 0.000 | 2/2 | 4.625 |
| MOVR | 0.444 | 1/1 | 0.000 | 0/0 | 9.562 |
| MUBARAK | 0.452 | 10/10 | 0.000 | 3/5 | 9.848 |
| NIGHT | 0.281 | 1/1 | 0.000 | 0/0 | 8.444 |
| NMR | 0.333 | 2/2 | 0.042 | 0/0 | 4.377 |
| NOM | 0.174 | 2/7 | 0.000 | 0/1 | 2.261 |
| PHA | 0.667 | 2/2 | 0.000 | 0/1 | 9.057 |
| QNT | 0.557 | 3/3 | 0.833 | 0/1 | 4.920 |
| SOON | 0.263 | 3/5 | 3.000 | 1/2 | 0.551 |
| TRUMP | 0.235 | 2/2 | 3.750 | 0/0 | 0.000 |
| ZEC | 0.488 | 7/7 | 0.500 | 4/4 | 17.077 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| oi_chg_24h | 0.360 | 30.787 |
| volume_z_7d | 1.443 | 24.756 |
| ret_24h | 0.664 | 9.801 |
| oi_z_24h | 0.450 | 6.601 |
| oi_chg_1h | 0.187 | 3.665 |
| ls_long_pct | -0.526 | 3.385 |
| oi_chg_4h | 0.181 | 2.668 |
| liq_short_4h_oi | -0.297 | 2.540 |
| buy_ratio_4h | 0.318 | 2.069 |
| funding_z_7d | 0.025 | 1.911 |
| ret_4h | -0.062 | 1.682 |
| volatility_24h | 0.063 | 1.245 |
| oi_z_7d | 0.094 | 1.179 |
| volume_ratio_24h | -0.040 | 1.131 |
| ret_1h | 0.172 | 1.076 |
| liq_long_4h_oi | -0.631 | 1.053 |
| oi_minus_ret_24h | -0.114 | 1.042 |
| liq_imbalance_4h | 0.210 | 1.020 |
| funding | 0.036 | 0.970 |
| funding_chg_24h | -0.246 | 0.605 |
| ls_chg_24h | 0.117 | 0.328 |
| range_1h | -0.234 | 0.285 |
| buy_ratio_1h | 0.105 | 0.201 |

## 15m model and signal tiers

LightGBM on 73588 15-minute bars (34 features: the hourly set with the same wall-clock windows plus short bar features), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 1.80 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.361 | 0.524 | 0.427 | 0.368 | 63/71 | 0.250 | 16/22 | 8.532 |
| 15m + 1h ≥ 0.75 (alert) | 0.388 | 0.483 | 0.430 | — | 61/71 | 0.583 | 11/22 | 5.949 |
