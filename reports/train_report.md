# Training report

Generated 2026-09-30 15:22 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 17810 hourly bars, 14 tickers (AGT, AIN, AKE, ARK, G, LSK, MOVR, MUBARAK, NIGHT, NMR, PHA, QNT, SOON, TRUMP), 394 pump hours (2.2%), 49 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.356 | 0.556 | 0.434 | — | 35/49 | 0.500 | 5/19 | 3.234 |
| logreg | 0.378 | 0.523 | 0.439 | 0.376 | 44/49 | 0.000 | 9/19 | 3.436 |
| lightgbm | 0.426 | 0.591 | 0.495 | 0.449 | 43/49 | 0.000 | 8/19 | 3.760 |

## Training settings

- `--fresh-hours 48`: 557 unlabeled hours after 2026-09-28 15:21 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.174 | 1/2 | 0.000 | 0/0 | 1.756 |
| AIN | 0.810 | 1/1 | 0.000 | 0/0 | 1.138 |
| AKE | 0.495 | 5/9 | 0.750 | 1/4 | 1.601 |
| ARK | 0.241 | 2/4 | 11.250 | 1/3 | 5.255 |
| G | 0.349 | 3/7 | 0.583 | 0/1 | 2.135 |
| LSK | 0.414 | 2/2 | 0.000 | 2/2 | 2.669 |
| MOVR | 0.429 | 1/1 | 0.500 | 0/0 | 7.524 |
| MUBARAK | 0.472 | 9/10 | 0.000 | 1/5 | 2.278 |
| NIGHT | 0.300 | 1/1 | 0.750 | 0/0 | 6.321 |
| NMR | 0.294 | 2/2 | 0.000 | 0/0 | 3.981 |
| PHA | 0.214 | 2/2 | 5.792 | 0/1 | 5.281 |
| QNT | 0.493 | 2/3 | 2.917 | 0/1 | 3.984 |
| SOON | 0.606 | 2/3 | 1.500 | 0/2 | 1.138 |
| TRUMP | 0.526 | 2/2 | 1.250 | 0/0 | 0.576 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.471 | 2/2 | 0.000 | 0/0 | 1.756 |
| AIN | 0.291 | 1/1 | 0.000 | 0/0 | 1.138 |
| AKE | 0.431 | 9/9 | 0.000 | 2/4 | 5.337 |
| ARK | 0.340 | 3/4 | 0.000 | 1/3 | 5.255 |
| G | 0.333 | 4/7 | 1.083 | 0/1 | 0.534 |
| LSK | 0.447 | 2/2 | 0.000 | 2/2 | 7.472 |
| MOVR | 0.667 | 1/1 | 0.000 | 0/0 | 4.051 |
| MUBARAK | 0.438 | 10/10 | 0.000 | 3/5 | 2.278 |
| NIGHT | 0.250 | 1/1 | 0.000 | 0/0 | 6.321 |
| NMR | 0.364 | 2/2 | 0.042 | 0/0 | 2.275 |
| PHA | 0.488 | 2/2 | 0.292 | 0/1 | 3.521 |
| QNT | 0.638 | 3/3 | 0.833 | 0/1 | 4.553 |
| SOON | 0.455 | 2/3 | 1.000 | 1/2 | 2.277 |
| TRUMP | 0.741 | 2/2 | 2.250 | 0/0 | 1.151 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.526 | 2/2 | 0.000 | 0/0 | 1.756 |
| AIN | 0.706 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.449 | 7/9 | 0.417 | 2/4 | 3.736 |
| ARK | 0.479 | 4/4 | 0.292 | 0/3 | 4.672 |
| G | 0.520 | 4/7 | 0.292 | 0/1 | 2.135 |
| LSK | 0.560 | 2/2 | 0.000 | 2/2 | 4.270 |
| MOVR | 0.627 | 1/1 | 0.000 | 0/0 | 4.630 |
| MUBARAK | 0.453 | 10/10 | 0.000 | 3/5 | 6.835 |
| NIGHT | 0.240 | 1/1 | 0.000 | 0/0 | 10.918 |
| NMR | 0.364 | 2/2 | 0.042 | 0/0 | 2.275 |
| PHA | 0.667 | 2/2 | 0.042 | 0/1 | 5.868 |
| QNT | 0.600 | 3/3 | 0.833 | 0/1 | 4.553 |
| SOON | 0.444 | 3/3 | 3.000 | 1/2 | 1.138 |
| TRUMP | 0.125 | 1/2 | 1.750 | 0/0 | 0.000 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| oi_chg_24h | 0.472 | 46.725 |
| ret_24h | 0.669 | 11.881 |
| volume_z_7d | 1.191 | 11.628 |
| oi_z_24h | 0.464 | 4.741 |
| ls_long_pct | -0.933 | 3.288 |
| ret_4h | 0.099 | 2.651 |
| oi_chg_4h | 0.108 | 2.194 |
| oi_z_7d | 0.781 | 1.687 |
| oi_minus_ret_24h | 0.025 | 1.646 |
| ls_chg_24h | 0.423 | 1.503 |
| liq_long_4h_oi | -1.294 | 1.497 |
| volatility_24h | -0.486 | 1.472 |
| oi_chg_1h | -0.109 | 1.333 |
| funding_z_7d | 0.024 | 1.238 |
| liq_short_4h_oi | -0.434 | 1.237 |
| volume_ratio_24h | 0.145 | 1.143 |
| buy_ratio_4h | 0.203 | 1.079 |
| ret_1h | 0.244 | 0.937 |
| funding_chg_24h | -0.232 | 0.813 |
| range_1h | 0.078 | 0.434 |
| funding | -0.205 | 0.397 |
| liq_imbalance_4h | -0.003 | 0.323 |
| buy_ratio_1h | 0.096 | 0.152 |

## 15m model and signal tiers

LightGBM on 60813 15-minute bars (34 features: the hourly set with the same wall-clock windows plus short bar features), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 1.16 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.462 | 0.591 | 0.518 | 0.453 | 46/49 | 0.208 | 11/19 | 5.304 |
| 15m + 1h ≥ 0.75 (alert) | 0.480 | 0.559 | 0.516 | — | 45/49 | 0.500 | 8/19 | 3.931 |
