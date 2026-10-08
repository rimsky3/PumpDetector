# Training report

Generated 2026-10-08 00:36 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 23503 hourly bars, 21 tickers (AGT, AIN, AKE, ARK, G, JTO, LSK, MAGIC, MON, MOVR, MUBARAK, NIGHT, NMR, NOM, PHA, QNT, SOON, TRUMP, VELVET, ZEC, ZRO), 670 pump hours (2.9%), 101 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.298 | 0.409 | 0.345 | — | 60/100 | 0.500 | 4/22 | 3.247 |
| logreg | 0.300 | 0.449 | 0.359 | 0.292 | 78/100 | 0.250 | 11/22 | 8.363 |
| lightgbm | 0.365 | 0.528 | 0.432 | 0.373 | 76/100 | 0.500 | 11/22 | 7.015 |

## Training settings

- `--fresh-hours 48`: 178 unlabeled hours after 2026-10-06 00:34 UTC left out of training
- models: rule, logreg, lightgbm; 20 features (without liquidations); saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.129 | 1/2 | 0.000 | 0/0 | 2.681 |
| AIN | 0.260 | 4/4 | 0.000 | 0/0 | 1.498 |
| AKE | 0.505 | 6/10 | 0.375 | 1/3 | 0.944 |
| ARK | 0.250 | 5/6 | 3.250 | 0/2 | 6.705 |
| G | 0.338 | 3/7 | 0.583 | 0/1 | 1.887 |
| JTO | 0.000 | 0/5 | — | 0/0 | 0.000 |
| LSK | 0.427 | 2/2 | 0.000 | 2/2 | 2.831 |
| MAGIC | 0.000 | 0/3 | — | 0/0 | 0.000 |
| MON | 0.133 | 1/3 | 15.250 | 0/0 | 12.743 |
| MOVR | 0.312 | 1/1 | 0.500 | 0/0 | 7.687 |
| MUBARAK | 0.390 | 9/10 | 0.000 | 1/5 | 3.997 |
| NIGHT | 0.344 | 5/12 | 2.750 | 0/0 | 1.578 |
| NMR | 0.172 | 2/2 | 0.000 | 0/0 | 3.997 |
| NOM | 0.071 | 2/7 | 0.000 | 0/1 | 1.629 |
| PHA | 0.194 | 2/2 | 5.792 | 0/1 | 5.128 |
| QNT | 0.444 | 3/4 | 2.000 | 0/1 | 5.492 |
| SOON | 0.495 | 3/5 | 0.000 | 0/2 | 1.006 |
| TRUMP | 0.500 | 2/2 | 1.250 | 0/0 | 1.008 |
| VELVET | 0.522 | 1/1 | 0.000 | 0/0 | 31.579 |
| ZEC | 0.609 | 5/7 | 0.500 | 0/4 | 1.006 |
| ZRO | 0.250 | 3/5 | 2.500 | 0/0 | 8.276 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.174 | 2/2 | 0.000 | 0/0 | 7.506 |
| AIN | 0.121 | 3/4 | 0.333 | 0/0 | 8.988 |
| AKE | 0.432 | 9/10 | 0.000 | 2/3 | 8.493 |
| ARK | 0.289 | 5/6 | 1.250 | 0/2 | 6.189 |
| G | 0.543 | 6/7 | 1.083 | 0/1 | 4.246 |
| JTO | 0.222 | 1/5 | 2.500 | 0/0 | 0.000 |
| LSK | 0.407 | 2/2 | 0.000 | 2/2 | 9.908 |
| MAGIC | 0.476 | 2/3 | 2.625 | 0/0 | 0.000 |
| MON | 0.167 | 1/3 | 15.250 | 0/0 | 12.743 |
| MOVR | 0.352 | 1/1 | 0.000 | 0/0 | 14.349 |
| MUBARAK | 0.432 | 10/10 | 0.000 | 3/5 | 9.993 |
| NIGHT | 0.172 | 8/12 | 2.500 | 0/0 | 2.630 |
| NMR | 0.174 | 2/2 | 0.042 | 0/0 | 6.995 |
| NOM | 0.179 | 4/7 | 0.000 | 0/1 | 7.602 |
| PHA | 0.556 | 2/2 | 0.042 | 0/1 | 8.718 |
| QNT | 0.583 | 4/4 | 0.000 | 0/1 | 6.990 |
| SOON | 0.311 | 3/5 | 3.000 | 1/2 | 3.017 |
| TRUMP | 0.522 | 2/2 | 0.375 | 0/0 | 7.059 |
| VELVET | 0.750 | 1/1 | 0.000 | 0/0 | 18.947 |
| ZEC | 0.422 | 7/7 | 0.250 | 3/4 | 22.123 |
| ZRO | 0.270 | 3/5 | 3.250 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.235 | 1/2 | 0.000 | 0/0 | 2.681 |
| AIN | 0.303 | 3/4 | 0.333 | 0/0 | 2.497 |
| AKE | 0.527 | 6/10 | 0.750 | 2/3 | 3.303 |
| ARK | 0.337 | 4/6 | 1.000 | 0/2 | 4.126 |
| G | 0.657 | 7/7 | 0.583 | 0/1 | 1.887 |
| JTO | 0.522 | 2/5 | 2.500 | 0/0 | 5.625 |
| LSK | 0.437 | 2/2 | 0.000 | 2/2 | 7.549 |
| MAGIC | 0.476 | 2/3 | 3.125 | 0/0 | 0.000 |
| MON | 0.353 | 1/3 | 9.250 | 0/0 | 31.858 |
| MOVR | 0.314 | 1/1 | 0.000 | 0/0 | 15.374 |
| MUBARAK | 0.395 | 10/10 | 0.000 | 3/5 | 8.994 |
| NIGHT | 0.438 | 9/12 | 2.750 | 0/0 | 4.207 |
| NMR | 0.129 | 2/2 | 0.042 | 0/0 | 9.493 |
| NOM | 0.194 | 3/7 | 0.000 | 0/1 | 3.258 |
| PHA | 0.489 | 2/2 | 0.000 | 0/1 | 11.282 |
| QNT | 0.432 | 4/4 | 0.417 | 0/1 | 11.983 |
| SOON | 0.500 | 3/5 | 2.167 | 1/2 | 3.017 |
| TRUMP | 0.516 | 2/2 | 2.250 | 0/0 | 3.025 |
| VELVET | 0.500 | 1/1 | 3.750 | 0/0 | 0.000 |
| ZEC | 0.472 | 7/7 | 0.500 | 3/4 | 16.089 |
| ZRO | 0.656 | 4/5 | 1.375 | 0/0 | 28.966 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| oi_z_24h | 0.734 | 19.760 |
| volume_z_7d | 0.983 | 18.760 |
| ret_24h | 0.509 | 13.287 |
| oi_chg_24h | 0.191 | 11.595 |
| ls_long_pct | -0.292 | 9.051 |
| volume_ratio_24h | -0.012 | 4.290 |
| funding | -0.036 | 3.608 |
| oi_chg_4h | 0.077 | 3.306 |
| ret_4h | 0.180 | 3.101 |
| oi_chg_1h | 0.158 | 2.823 |
| funding_z_7d | 0.028 | 2.509 |
| funding_chg_24h | -0.092 | 1.570 |
| buy_ratio_4h | 0.267 | 1.229 |
| ret_1h | 0.192 | 1.186 |
| oi_z_7d | 0.178 | 1.044 |
| oi_minus_ret_24h | -0.215 | 0.985 |
| volatility_24h | -0.202 | 0.824 |
| range_1h | -0.584 | 0.515 |
| ls_chg_24h | 0.036 | 0.449 |
| buy_ratio_1h | 0.020 | 0.109 |

### LightGBM gain by feature group

Correlated features share gain and can swap places between runs; the group total is more stable.

| group | gain_% | top feature |
|---|---|---|
| OI | 39.500 | oi_z_24h (19.8%) |
| Обсяг | 23.000 | volume_z_7d (18.8%) |
| Ціна | 17.600 | ret_24h (13.3%) |
| Long / Short | 9.500 | ls_long_pct (9.1%) |
| Funding | 7.700 | funding (3.6%) |
| Волатильність | 1.300 | volatility_24h (0.8%) |
| Купівлі / продажі | 1.300 | buy_ratio_4h (1.2%) |

## 15m model and signal tiers

LightGBM on 94725 15-minute bars (41 features: the hourly set with the same wall-clock windows plus short bar features and own-move features vs the ALT-EW index — present on 100% of bars), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 2.05 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.322 | 0.501 | 0.392 | 0.348 | 83/100 | 0.667 | 17/22 | 9.456 |
| 15m + 1h ≥ 0.75 (alert) | 0.353 | 0.471 | 0.404 | — | 81/100 | 0.833 | 15/22 | 6.506 |

### 15m LightGBM gain by feature group

| group | gain_% | top feature |
|---|---|---|
| OI | 39.100 | oi_z_24h (20.4%) |
| Власний рух | 17.200 | r2 (5.7%) |
| Ціна | 16.300 | ret_24h_vol (11.3%) |
| Обсяг | 10.900 | volume_z_7d (8.4%) |
| Long / Short | 6.900 | ls_long_pct (6.7%) |
| Funding | 4.200 | funding_z_7d (3.1%) |
| Волатильність | 4.000 | volatility_24h (1.7%) |
| Ліквідність (Amihud) | 1.100 | illiq_x (1.1%) |
| Купівлі / продажі | 0.200 | buy_ratio_4h (0.2%) |
