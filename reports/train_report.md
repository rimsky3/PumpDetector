# Training report

Generated 2026-10-01 23:56 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 20786 hourly bars, 16 tickers (AGT, AIN, AKE, ARK, G, LSK, MOVR, MUBARAK, NIGHT, NMR, NOM, PHA, QNT, SOON, TRUMP, ZEC), 435 pump hours (2.1%), 72 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.304 | 0.492 | 0.376 | — | 48/71 | 0.083 | 5/22 | 3.014 |
| logreg | 0.365 | 0.510 | 0.425 | 0.335 | 58/71 | 0.042 | 10/22 | 4.157 |
| lightgbm | 0.391 | 0.517 | 0.445 | 0.394 | 56/71 | 0.000 | 12/22 | 4.746 |

## Training settings

- `--fresh-hours 48`: 669 unlabeled hours after 2026-09-29 23:55 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.182 | 1/2 | 0.000 | 0/0 | 1.713 |
| AIN | 0.400 | 4/4 | 0.000 | 0/0 | 1.110 |
| AKE | 0.495 | 5/10 | 0.750 | 1/3 | 1.564 |
| ARK | 0.224 | 4/6 | 2.750 | 0/2 | 4.553 |
| G | 0.349 | 3/7 | 0.583 | 0/1 | 2.085 |
| LSK | 0.414 | 2/2 | 0.000 | 2/2 | 2.607 |
| MOVR | 0.429 | 1/1 | 0.500 | 0/0 | 7.370 |
| MUBARAK | 0.414 | 9/10 | 0.000 | 2/5 | 4.444 |
| NIGHT | 0.300 | 1/1 | 0.750 | 0/0 | 6.231 |
| NMR | 0.200 | 2/2 | 0.000 | 0/0 | 3.889 |
| NOM | 0.091 | 2/7 | 0.000 | 0/1 | 1.144 |
| PHA | 0.214 | 2/2 | 5.792 | 0/1 | 5.147 |
| QNT | 0.493 | 2/3 | 2.917 | 0/1 | 3.886 |
| SOON | 0.454 | 3/5 | 0.000 | 0/2 | 1.106 |
| TRUMP | 0.526 | 2/2 | 1.250 | 0/0 | 0.561 |
| ZEC | 0.523 | 5/7 | 0.500 | 0/4 | 1.119 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.400 | 2/2 | 0.000 | 0/0 | 3.426 |
| AIN | 0.118 | 2/4 | 2.167 | 0/0 | 3.331 |
| AKE | 0.381 | 9/10 | 0.000 | 2/3 | 10.427 |
| ARK | 0.289 | 4/6 | 2.250 | 0/2 | 5.692 |
| G | 0.462 | 5/7 | 1.583 | 0/1 | 0.521 |
| LSK | 0.506 | 2/2 | 0.000 | 2/2 | 5.214 |
| MOVR | 0.585 | 1/1 | 0.000 | 0/0 | 4.535 |
| MUBARAK | 0.492 | 10/10 | 0.000 | 2/5 | 2.778 |
| NIGHT | 0.270 | 1/1 | 0.000 | 0/0 | 2.832 |
| NMR | 0.296 | 2/2 | 0.042 | 0/0 | 3.889 |
| NOM | 0.171 | 3/7 | 0.000 | 0/1 | 3.431 |
| PHA | 0.615 | 2/2 | 0.042 | 0/1 | 5.719 |
| QNT | 0.645 | 3/3 | 0.833 | 0/1 | 4.996 |
| SOON | 0.282 | 3/5 | 3.000 | 1/2 | 1.659 |
| TRUMP | 0.750 | 2/2 | 2.250 | 0/0 | 0.000 |
| ZEC | 0.511 | 7/7 | 0.500 | 3/4 | 7.832 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.500 | 2/2 | 0.000 | 0/0 | 1.142 |
| AIN | 0.200 | 3/4 | 0.583 | 0/0 | 0.555 |
| AKE | 0.435 | 7/10 | 0.167 | 2/3 | 4.171 |
| ARK | 0.430 | 5/6 | 1.250 | 0/2 | 4.553 |
| G | 0.538 | 5/7 | 0.583 | 0/1 | 2.085 |
| LSK | 0.419 | 2/2 | 0.000 | 2/2 | 4.692 |
| MOVR | 0.653 | 1/1 | 0.000 | 0/0 | 7.370 |
| MUBARAK | 0.458 | 10/10 | 0.000 | 3/5 | 10.000 |
| NIGHT | 0.247 | 1/1 | 0.000 | 0/0 | 11.330 |
| NMR | 0.421 | 2/2 | 0.042 | 0/0 | 3.333 |
| NOM | 0.200 | 2/7 | 0.000 | 0/1 | 1.144 |
| PHA | 0.667 | 2/2 | 0.000 | 0/1 | 8.578 |
| QNT | 0.559 | 3/3 | 1.833 | 0/1 | 3.886 |
| SOON | 0.381 | 3/5 | 2.167 | 1/2 | 1.106 |
| TRUMP | 0.235 | 2/2 | 3.750 | 0/0 | 0.000 |
| ZEC | 0.474 | 6/7 | 0.375 | 4/4 | 12.308 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| oi_chg_24h | 0.378 | 32.284 |
| volume_z_7d | 1.443 | 24.593 |
| ret_24h | 0.677 | 8.564 |
| oi_z_24h | 0.447 | 5.863 |
| oi_chg_1h | 0.173 | 3.965 |
| ls_long_pct | -0.526 | 3.345 |
| oi_chg_4h | 0.177 | 2.931 |
| liq_short_4h_oi | -0.328 | 2.599 |
| buy_ratio_4h | 0.313 | 2.254 |
| funding_z_7d | 0.025 | 1.874 |
| ret_4h | -0.037 | 1.600 |
| volume_ratio_24h | -0.049 | 1.326 |
| oi_z_7d | 0.077 | 1.226 |
| liq_long_4h_oi | -0.617 | 1.208 |
| ret_1h | 0.182 | 1.094 |
| volatility_24h | 0.050 | 0.991 |
| oi_minus_ret_24h | -0.106 | 0.986 |
| liq_imbalance_4h | 0.212 | 0.977 |
| funding | 0.049 | 0.843 |
| funding_chg_24h | -0.257 | 0.482 |
| range_1h | -0.230 | 0.482 |
| ls_chg_24h | 0.096 | 0.340 |
| buy_ratio_1h | 0.107 | 0.175 |

## 15m model and signal tiers

LightGBM on 72537 15-minute bars (34 features: the hourly set with the same wall-clock windows plus short bar features), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 1.80 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.410 | 0.510 | 0.455 | 0.382 | 62/71 | 0.458 | 14/22 | 6.948 |
| 15m + 1h ≥ 0.75 (alert) | 0.433 | 0.472 | 0.452 | — | 61/71 | 0.917 | 9/22 | 5.281 |
