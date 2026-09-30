# Training report

Generated 2026-09-30 19:37 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 19093 hourly bars, 15 tickers (AGT, AIN, AKE, ARK, G, LSK, MOVR, MUBARAK, NIGHT, NMR, NOM, PHA, QNT, SOON, TRUMP), 404 pump hours (2.1%), 56 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.339 | 0.547 | 0.419 | — | 37/56 | 0.167 | 5/20 | 3.130 |
| logreg | 0.370 | 0.547 | 0.442 | 0.362 | 49/56 | 0.000 | 9/20 | 3.960 |
| lightgbm | 0.436 | 0.589 | 0.501 | 0.446 | 44/56 | 0.000 | 8/20 | 3.281 |

## Training settings

- `--fresh-hours 48`: 601 unlabeled hours after 2026-09-28 19:36 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.182 | 1/2 | 0.000 | 0/0 | 1.752 |
| AIN | 0.810 | 1/1 | 0.000 | 0/0 | 1.135 |
| AKE | 0.495 | 5/9 | 0.750 | 1/4 | 1.596 |
| ARK | 0.241 | 2/4 | 11.250 | 1/3 | 5.238 |
| G | 0.349 | 3/7 | 0.583 | 0/1 | 2.129 |
| LSK | 0.414 | 2/2 | 0.000 | 2/2 | 2.661 |
| MOVR | 0.429 | 1/1 | 0.500 | 0/0 | 7.500 |
| MUBARAK | 0.472 | 9/10 | 0.000 | 1/5 | 2.271 |
| NIGHT | 0.300 | 1/1 | 0.750 | 0/0 | 6.301 |
| NMR | 0.270 | 2/2 | 0.000 | 0/0 | 4.535 |
| NOM | 0.091 | 2/7 | 0.000 | 0/1 | 1.169 |
| PHA | 0.214 | 2/2 | 5.792 | 0/1 | 5.264 |
| QNT | 0.493 | 2/3 | 2.917 | 0/1 | 3.972 |
| SOON | 0.606 | 2/3 | 1.500 | 0/2 | 1.138 |
| TRUMP | 0.526 | 2/2 | 1.250 | 0/0 | 0.574 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.444 | 2/2 | 0.000 | 0/0 | 2.920 |
| AIN | 0.259 | 1/1 | 0.000 | 0/0 | 1.702 |
| AKE | 0.429 | 9/9 | 0.000 | 2/4 | 5.322 |
| ARK | 0.436 | 3/4 | 0.000 | 1/3 | 5.821 |
| G | 0.448 | 6/7 | 1.083 | 0/1 | 0.532 |
| LSK | 0.452 | 2/2 | 0.000 | 2/2 | 6.386 |
| MOVR | 0.609 | 1/1 | 0.000 | 0/0 | 5.769 |
| MUBARAK | 0.449 | 10/10 | 0.000 | 3/5 | 2.839 |
| NIGHT | 0.238 | 1/1 | 0.000 | 0/0 | 5.155 |
| NMR | 0.267 | 2/2 | 0.042 | 0/0 | 4.535 |
| NOM | 0.194 | 3/7 | 0.000 | 0/1 | 2.922 |
| PHA | 0.500 | 2/2 | 0.042 | 0/1 | 8.188 |
| QNT | 0.611 | 3/3 | 0.833 | 0/1 | 5.106 |
| SOON | 0.431 | 2/3 | 1.000 | 1/2 | 2.277 |
| TRUMP | 0.800 | 2/2 | 2.250 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.154 | 1/2 | 0.750 | 0/0 | 1.752 |
| AIN | 0.706 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.523 | 8/9 | 0.250 | 2/4 | 2.661 |
| ARK | 0.492 | 4/4 | 0.292 | 0/3 | 5.238 |
| G | 0.542 | 4/7 | 0.292 | 0/1 | 1.064 |
| LSK | 0.582 | 2/2 | 0.000 | 2/2 | 4.257 |
| MOVR | 0.615 | 1/1 | 0.000 | 0/0 | 5.769 |
| MUBARAK | 0.468 | 10/10 | 0.000 | 3/5 | 7.382 |
| NIGHT | 0.278 | 1/1 | 0.000 | 0/0 | 8.592 |
| NMR | 0.381 | 2/2 | 0.042 | 0/0 | 1.701 |
| NOM | 0.222 | 2/7 | 0.000 | 0/1 | 0.584 |
| PHA | 0.632 | 2/2 | 0.042 | 0/1 | 5.849 |
| QNT | 0.620 | 3/3 | 0.833 | 0/1 | 3.972 |
| SOON | 0.408 | 3/3 | 3.000 | 1/2 | 0.569 |
| TRUMP | 0.000 | 0/2 | — | 0/0 | 0.000 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| oi_chg_24h | 0.497 | 46.222 |
| volume_z_7d | 1.099 | 11.083 |
| ret_24h | 0.737 | 10.451 |
| oi_z_24h | 0.570 | 5.878 |
| ls_long_pct | -0.936 | 3.976 |
| ret_4h | 0.114 | 2.803 |
| oi_chg_4h | 0.057 | 2.548 |
| volatility_24h | -0.573 | 1.828 |
| oi_minus_ret_24h | 0.001 | 1.698 |
| ret_1h | 0.207 | 1.627 |
| oi_z_7d | 0.656 | 1.607 |
| oi_chg_1h | -0.038 | 1.605 |
| ls_chg_24h | 0.485 | 1.580 |
| liq_long_4h_oi | -1.248 | 1.556 |
| funding_z_7d | 0.023 | 1.219 |
| liq_short_4h_oi | -0.443 | 1.046 |
| buy_ratio_4h | 0.131 | 0.868 |
| volume_ratio_24h | 0.169 | 0.696 |
| funding_chg_24h | -0.219 | 0.525 |
| funding | -0.206 | 0.408 |
| liq_imbalance_4h | 0.008 | 0.303 |
| range_1h | 0.183 | 0.276 |
| buy_ratio_1h | 0.111 | 0.197 |

## 15m model and signal tiers

LightGBM on 65957 15-minute bars (34 features: the hourly set with the same wall-clock windows plus short bar features), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 2.48 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.448 | 0.563 | 0.499 | 0.444 | 48/56 | 0.042 | 13/20 | 6.069 |
| 15m + 1h ≥ 0.75 (alert) | 0.470 | 0.532 | 0.499 | — | 48/56 | 0.500 | 8/20 | 4.323 |
