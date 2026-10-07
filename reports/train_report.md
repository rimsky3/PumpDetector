# Training report

Generated 2026-10-07 13:58 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 23347 hourly bars, 21 tickers (AGT, AIN, AKE, ARK, G, JTO, LSK, MAGIC, MON, MOVR, MUBARAK, NIGHT, NMR, NOM, PHA, QNT, SOON, TRUMP, VELVET, ZEC, ZRO), 670 pump hours (2.9%), 101 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.298 | 0.409 | 0.345 | — | 60/100 | 0.500 | 4/22 | 3.269 |
| logreg | 0.302 | 0.445 | 0.359 | 0.292 | 79/100 | 0.333 | 12/22 | 8.203 |
| lightgbm | 0.358 | 0.525 | 0.426 | 0.369 | 78/100 | 0.292 | 10/22 | 7.432 |

## Training settings

- `--fresh-hours 48`: 334 unlabeled hours after 2026-10-05 13:56 UTC left out of training
- models: rule, logreg, lightgbm; 20 features (without liquidations); saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.129 | 1/2 | 0.000 | 0/0 | 2.681 |
| AIN | 0.260 | 4/4 | 0.000 | 0/0 | 1.509 |
| AKE | 0.505 | 6/10 | 0.375 | 1/3 | 0.950 |
| ARK | 0.250 | 5/6 | 3.250 | 0/2 | 6.714 |
| G | 0.338 | 3/7 | 0.583 | 0/1 | 1.901 |
| JTO | 0.000 | 0/5 | — | 0/0 | 0.000 |
| LSK | 0.427 | 2/2 | 0.000 | 2/2 | 2.851 |
| MAGIC | 0.000 | 0/3 | — | 0/0 | 0.000 |
| MON | 0.133 | 1/3 | 15.250 | 0/0 | 12.743 |
| MOVR | 0.312 | 1/1 | 0.500 | 0/0 | 7.747 |
| MUBARAK | 0.390 | 9/10 | 0.000 | 1/5 | 4.028 |
| NIGHT | 0.344 | 5/12 | 2.750 | 0/0 | 1.578 |
| NMR | 0.172 | 2/2 | 0.000 | 0/0 | 4.028 |
| NOM | 0.071 | 2/7 | 0.000 | 0/1 | 1.629 |
| PHA | 0.194 | 2/2 | 5.792 | 0/1 | 5.169 |
| QNT | 0.444 | 3/4 | 2.000 | 0/1 | 5.535 |
| SOON | 0.495 | 3/5 | 0.000 | 0/2 | 1.013 |
| TRUMP | 0.500 | 2/2 | 1.250 | 0/0 | 1.016 |
| VELVET | 0.522 | 1/1 | 0.000 | 0/0 | 31.579 |
| ZEC | 0.609 | 5/7 | 0.500 | 0/4 | 1.013 |
| ZRO | 0.250 | 3/5 | 2.500 | 0/0 | 8.834 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.174 | 2/2 | 0.000 | 0/0 | 7.506 |
| AIN | 0.127 | 3/4 | 0.333 | 0/0 | 8.553 |
| AKE | 0.432 | 9/10 | 0.000 | 2/3 | 8.554 |
| ARK | 0.289 | 5/6 | 1.250 | 0/2 | 6.198 |
| G | 0.543 | 6/7 | 1.083 | 0/1 | 4.277 |
| JTO | 0.222 | 1/5 | 2.500 | 0/0 | 0.000 |
| LSK | 0.410 | 2/2 | 0.000 | 2/2 | 9.505 |
| MAGIC | 0.400 | 2/3 | 3.125 | 0/0 | 0.000 |
| MON | 0.167 | 1/3 | 15.250 | 0/0 | 12.743 |
| MOVR | 0.360 | 1/1 | 0.000 | 0/0 | 13.945 |
| MUBARAK | 0.429 | 10/10 | 0.000 | 3/5 | 8.559 |
| NIGHT | 0.172 | 8/12 | 2.500 | 0/0 | 2.630 |
| NMR | 0.174 | 2/2 | 0.042 | 0/0 | 7.049 |
| NOM | 0.179 | 4/7 | 0.000 | 0/1 | 7.602 |
| PHA | 0.556 | 2/2 | 0.042 | 0/1 | 8.787 |
| QNT | 0.571 | 4/4 | 0.000 | 1/1 | 7.547 |
| SOON | 0.292 | 3/5 | 3.000 | 1/2 | 3.040 |
| TRUMP | 0.533 | 2/2 | 0.375 | 0/0 | 6.606 |
| VELVET | 0.800 | 1/1 | 0.000 | 0/0 | 12.632 |
| ZEC | 0.417 | 7/7 | 0.250 | 3/4 | 22.294 |
| ZRO | 0.359 | 4/5 | 2.875 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.261 | 2/2 | 0.000 | 0/0 | 4.825 |
| AIN | 0.343 | 4/4 | 0.167 | 0/0 | 1.509 |
| AKE | 0.506 | 6/10 | 0.750 | 1/3 | 2.851 |
| ARK | 0.343 | 4/6 | 1.250 | 0/2 | 4.132 |
| G | 0.667 | 7/7 | 0.583 | 0/1 | 1.901 |
| JTO | 0.522 | 2/5 | 2.500 | 0/0 | 6.154 |
| LSK | 0.455 | 2/2 | 0.000 | 2/2 | 9.505 |
| MAGIC | 0.211 | 1/3 | 4.000 | 0/0 | 0.000 |
| MON | 0.343 | 1/3 | 9.250 | 0/0 | 25.487 |
| MOVR | 0.305 | 1/1 | 0.000 | 0/0 | 15.495 |
| MUBARAK | 0.389 | 10/10 | 0.000 | 3/5 | 10.070 |
| NIGHT | 0.440 | 10/12 | 2.750 | 0/0 | 4.733 |
| NMR | 0.121 | 2/2 | 0.042 | 0/0 | 10.070 |
| NOM | 0.200 | 3/7 | 0.000 | 0/1 | 3.801 |
| PHA | 0.494 | 2/2 | 0.000 | 0/1 | 11.371 |
| QNT | 0.475 | 4/4 | 0.000 | 0/1 | 12.075 |
| SOON | 0.454 | 3/5 | 2.167 | 1/2 | 3.040 |
| TRUMP | 0.439 | 2/2 | 2.250 | 0/0 | 5.589 |
| VELVET | 0.500 | 1/1 | 3.750 | 0/0 | 0.000 |
| ZEC | 0.466 | 7/7 | 0.500 | 3/4 | 16.214 |
| ZRO | 0.678 | 4/5 | 1.375 | 0/0 | 22.086 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| volume_z_7d | 0.980 | 21.489 |
| oi_z_24h | 0.735 | 18.871 |
| ret_24h | 0.509 | 13.304 |
| ls_long_pct | -0.297 | 9.126 |
| oi_chg_24h | 0.190 | 8.084 |
| oi_chg_4h | 0.077 | 4.963 |
| volume_ratio_24h | -0.011 | 4.240 |
| funding | -0.038 | 3.411 |
| oi_chg_1h | 0.160 | 2.812 |
| ret_4h | 0.180 | 2.730 |
| funding_z_7d | 0.027 | 2.448 |
| funding_chg_24h | -0.091 | 1.673 |
| oi_z_7d | 0.180 | 1.522 |
| ret_1h | 0.189 | 1.222 |
| buy_ratio_4h | 0.266 | 1.065 |
| oi_minus_ret_24h | -0.217 | 0.950 |
| volatility_24h | -0.202 | 0.905 |
| range_1h | -0.585 | 0.598 |
| ls_chg_24h | 0.038 | 0.396 |
| buy_ratio_1h | 0.020 | 0.192 |

### LightGBM gain by feature group

Correlated features share gain and can swap places between runs; the group total is more stable.

| group | gain_% | top feature |
|---|---|---|
| OI | 37.200 | oi_z_24h (18.9%) |
| Обсяг | 25.700 | volume_z_7d (21.5%) |
| Ціна | 17.300 | ret_24h (13.3%) |
| Long / Short | 9.500 | ls_long_pct (9.1%) |
| Funding | 7.500 | funding (3.4%) |
| Волатильність | 1.500 | volatility_24h (0.9%) |
| Купівлі / продажі | 1.300 | buy_ratio_4h (1.1%) |

## 15m model and signal tiers

LightGBM on 83886 15-minute bars (29 features: the hourly set with the same wall-clock windows plus short bar features), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 1.91 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.355 | 0.499 | 0.415 | 0.367 | 84/100 | 0.500 | 15/22 | 10.918 |
| 15m + 1h ≥ 0.75 (alert) | 0.385 | 0.470 | 0.423 | — | 80/100 | 0.667 | 13/22 | 7.347 |

### 15m LightGBM gain by feature group

| group | gain_% | top feature |
|---|---|---|
| OI | 52.400 | oi_z_24h (36.5%) |
| Обсяг | 16.100 | volume_z_7d (9.2%) |
| Ціна | 13.000 | ret_24h (8.1%) |
| Funding | 9.600 | funding_z_7d (6.7%) |
| Long / Short | 5.100 | ls_long_pct (4.3%) |
| Волатильність | 3.300 | volatility_24h (3.1%) |
| Купівлі / продажі | 0.500 | buy_ratio_4h (0.5%) |
