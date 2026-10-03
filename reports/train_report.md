# Training report

Generated 2026-10-03 12:07 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 21485 hourly bars, 18 tickers (AGT, AIN, AKE, ARK, G, LSK, MON, MOVR, MUBARAK, NIGHT, NMR, NOM, PHA, QNT, SOON, TRUMP, ZEC, ZRO), 632 pump hours (2.9%), 92 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.306 | 0.419 | 0.354 | — | 60/91 | 0.542 | 5/22 | 3.318 |
| logreg | 0.248 | 0.449 | 0.320 | 0.306 | 72/91 | 0.000 | 13/22 | 8.445 |
| lightgbm | 0.391 | 0.521 | 0.447 | 0.405 | 71/91 | 0.250 | 10/22 | 5.831 |

## Training settings

- `--fresh-hours 48`: 714 unlabeled hours after 2026-10-01 12:06 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.129 | 1/2 | 0.000 | 0/0 | 2.778 |
| AIN | 0.364 | 4/4 | 0.000 | 0/0 | 1.079 |
| AKE | 0.505 | 6/10 | 0.375 | 1/3 | 1.016 |
| ARK | 0.250 | 5/6 | 3.250 | 0/2 | 7.217 |
| G | 0.338 | 3/7 | 0.583 | 0/1 | 2.031 |
| LSK | 0.427 | 2/2 | 0.000 | 2/2 | 3.047 |
| MON | 0.133 | 1/3 | 15.250 | 0/0 | 16.552 |
| MOVR | 0.312 | 1/1 | 0.500 | 0/0 | 8.327 |
| MUBARAK | 0.429 | 9/10 | 0.000 | 1/5 | 2.701 |
| NIGHT | 0.344 | 5/12 | 2.750 | 0/0 | 1.653 |
| NMR | 0.172 | 2/2 | 0.000 | 0/0 | 4.321 |
| NOM | 0.071 | 2/7 | 0.000 | 0/1 | 1.673 |
| PHA | 0.194 | 2/2 | 5.792 | 0/1 | 5.660 |
| QNT | 0.444 | 3/4 | 2.000 | 0/1 | 5.937 |
| SOON | 0.495 | 3/5 | 0.000 | 0/2 | 1.088 |
| TRUMP | 0.500 | 2/2 | 1.250 | 0/0 | 1.091 |
| ZEC | 0.500 | 6/7 | 0.625 | 1/4 | 3.263 |
| ZRO | 0.235 | 3/5 | 2.500 | 0/0 | 0.000 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.125 | 2/2 | 0.000 | 0/0 | 8.889 |
| AIN | 0.093 | 2/4 | 2.167 | 0/0 | 4.318 |
| AKE | 0.359 | 9/10 | 0.000 | 2/3 | 11.678 |
| ARK | 0.294 | 5/6 | 3.250 | 0/2 | 7.772 |
| G | 0.542 | 5/7 | 1.583 | 0/1 | 2.539 |
| LSK | 0.374 | 2/2 | 0.000 | 2/2 | 11.678 |
| MON | 0.167 | 1/3 | 15.250 | 0/0 | 16.552 |
| MOVR | 0.414 | 1/1 | 0.000 | 0/0 | 8.327 |
| MUBARAK | 0.409 | 10/10 | 0.000 | 3/5 | 9.182 |
| NIGHT | 0.148 | 8/12 | 4.000 | 0/0 | 0.551 |
| NMR | 0.267 | 2/2 | 0.042 | 0/0 | 4.861 |
| NOM | 0.164 | 4/7 | 0.000 | 0/1 | 6.692 |
| PHA | 0.519 | 2/2 | 0.042 | 0/1 | 6.792 |
| QNT | 0.475 | 4/4 | 0.000 | 1/1 | 13.493 |
| SOON | 0.311 | 3/5 | 3.000 | 1/2 | 3.263 |
| TRUMP | 0.579 | 2/2 | 0.375 | 0/0 | 5.455 |
| ZEC | 0.281 | 7/7 | 0.000 | 4/4 | 29.366 |
| ZRO | 0.182 | 3/5 | 3.250 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.235 | 1/2 | 0.000 | 0/0 | 2.778 |
| AIN | 0.222 | 3/4 | 0.583 | 0/0 | 1.619 |
| AKE | 0.500 | 6/10 | 0.750 | 1/3 | 3.554 |
| ARK | 0.397 | 5/6 | 1.250 | 0/2 | 4.996 |
| G | 0.794 | 7/7 | 0.000 | 0/1 | 1.523 |
| LSK | 0.380 | 2/2 | 0.000 | 2/2 | 6.093 |
| MON | 0.387 | 1/3 | 9.250 | 0/0 | 16.552 |
| MOVR | 0.416 | 1/1 | 0.000 | 0/0 | 12.768 |
| MUBARAK | 0.419 | 10/10 | 0.000 | 3/5 | 6.482 |
| NIGHT | 0.479 | 9/12 | 2.750 | 0/0 | 1.653 |
| NMR | 0.148 | 2/2 | 0.042 | 0/0 | 8.642 |
| NOM | 0.148 | 2/7 | 0.000 | 0/1 | 1.673 |
| PHA | 0.643 | 2/2 | 0.542 | 0/1 | 6.226 |
| QNT | 0.462 | 4/4 | 0.417 | 0/1 | 11.334 |
| SOON | 0.396 | 3/5 | 2.167 | 1/2 | 2.175 |
| TRUMP | 0.364 | 2/2 | 3.250 | 0/0 | 1.636 |
| ZEC | 0.430 | 7/7 | 0.500 | 3/4 | 20.121 |
| ZRO | 0.776 | 4/5 | 1.625 | 0/0 | 0.000 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.523 | 16.784 |
| volume_z_7d | 0.970 | 15.210 |
| oi_z_24h | 0.661 | 14.991 |
| oi_chg_24h | 0.250 | 14.334 |
| ls_long_pct | -0.289 | 9.069 |
| oi_chg_4h | 0.153 | 3.766 |
| volume_ratio_24h | -0.007 | 3.161 |
| liq_short_4h_oi | -0.125 | 2.826 |
| ret_4h | 0.017 | 2.773 |
| funding_z_7d | 0.028 | 2.524 |
| funding | -0.050 | 2.065 |
| oi_chg_1h | 0.015 | 1.866 |
| buy_ratio_4h | 0.305 | 1.656 |
| oi_z_7d | 0.250 | 1.458 |
| liq_long_4h_oi | -0.734 | 1.456 |
| funding_chg_24h | -0.139 | 1.321 |
| ret_1h | 0.259 | 1.291 |
| oi_minus_ret_24h | -0.138 | 1.023 |
| volatility_24h | -0.023 | 0.845 |
| liq_imbalance_4h | 0.167 | 0.730 |
| range_1h | -0.276 | 0.387 |
| ls_chg_24h | 0.069 | 0.246 |
| buy_ratio_1h | 0.053 | 0.219 |

## 15m model and signal tiers

LightGBM on 75371 15-minute bars (34 features: the hourly set with the same wall-clock windows plus short bar features), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 1.57 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.362 | 0.482 | 0.413 | 0.376 | 79/91 | 0.500 | 16/22 | 10.126 |
| 15m + 1h ≥ 0.75 (alert) | 0.393 | 0.457 | 0.422 | — | 76/91 | 0.792 | 14/22 | 7.375 |
