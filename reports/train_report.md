# Training report

Generated 2026-10-10 23:35 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 30861 hourly bars, 25 tickers (AGT, AIN, AKE, API3, ARK, ENA, G, JTO, LSK, MAGIC, MON, MOVR, MUBARAK, NIGHT, NMR, NOM, ORCA, PHA, QNT, RLC, SOON, TRUMP, VELVET, ZEC, ZRO), 861 pump hours (2.8%), 131 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.286 | 0.395 | 0.332 | — | 78/130 | 0.375 | 6/27 | 3.570 |
| logreg | 0.290 | 0.431 | 0.347 | 0.302 | 105/130 | 0.500 | 13/27 | 8.352 |
| lightgbm | 0.341 | 0.524 | 0.413 | 0.345 | 109/130 | 0.500 | 14/27 | 8.562 |

## Training settings

- `--fresh-hours 48`: 0 unlabeled hours after 2026-10-08 23:32 UTC left out of training
- models: rule, logreg, lightgbm; 20 features (without liquidations); saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.129 | 1/2 | 0.000 | 0/0 | 2.681 |
| AIN | 0.368 | 6/7 | 0.000 | 0/0 | 0.990 |
| AKE | 0.505 | 6/10 | 0.375 | 1/3 | 0.936 |
| API3 | 0.372 | 2/6 | 0.000 | 1/4 | 0.000 |
| ARK | 0.250 | 5/6 | 3.250 | 0/2 | 6.705 |
| ENA | 0.423 | 2/4 | 0.750 | 0/0 | 0.000 |
| G | 0.338 | 3/7 | 0.583 | 0/1 | 1.870 |
| JTO | 0.000 | 0/5 | — | 0/0 | 1.698 |
| LSK | 0.427 | 2/2 | 0.000 | 2/2 | 2.805 |
| MAGIC | 0.041 | 1/3 | 0.500 | 0/0 | 2.264 |
| MON | 0.090 | 2/3 | 9.500 | 0/0 | 6.792 |
| MOVR | 0.312 | 1/1 | 0.500 | 0/0 | 7.622 |
| MUBARAK | 0.390 | 9/10 | 0.000 | 1/5 | 3.959 |
| NIGHT | 0.344 | 5/12 | 2.750 | 0/0 | 1.578 |
| NMR | 0.257 | 3/4 | 0.000 | 0/0 | 3.959 |
| NOM | 0.071 | 2/7 | 0.000 | 0/1 | 1.629 |
| ORCA | 0.216 | 6/13 | 0.750 | 1/1 | 3.962 |
| PHA | 0.194 | 2/2 | 5.792 | 0/1 | 5.085 |
| QNT | 0.444 | 3/4 | 2.000 | 0/1 | 5.440 |
| RLC | 0.581 | 2/2 | 0.125 | 0/0 | 4.528 |
| SOON | 0.495 | 3/5 | 0.000 | 0/2 | 0.997 |
| TRUMP | 0.500 | 2/2 | 1.250 | 0/0 | 1.000 |
| VELVET | 0.522 | 1/1 | 0.000 | 0/0 | 31.579 |
| ZEC | 0.609 | 5/7 | 0.500 | 0/4 | 0.997 |
| ZRO | 0.157 | 4/5 | 3.125 | 0/0 | 10.189 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.157 | 2/2 | 0.000 | 0/0 | 8.578 |
| AIN | 0.302 | 6/7 | 0.000 | 0/0 | 7.428 |
| AKE | 0.375 | 9/10 | 0.000 | 2/3 | 10.767 |
| API3 | 0.372 | 3/6 | 0.750 | 1/4 | 0.000 |
| ARK | 0.280 | 5/6 | 1.250 | 0/2 | 6.705 |
| ENA | 0.327 | 4/4 | 1.125 | 0/0 | 0.000 |
| G | 0.535 | 7/7 | 0.500 | 0/1 | 4.675 |
| JTO | 0.150 | 2/5 | 1.125 | 0/0 | 5.660 |
| LSK | 0.381 | 2/2 | 0.000 | 2/2 | 10.286 |
| MAGIC | 0.222 | 3/3 | 2.250 | 0/0 | 6.792 |
| MON | 0.140 | 2/3 | 9.000 | 0/0 | 10.189 |
| MOVR | 0.344 | 1/1 | 0.000 | 0/0 | 12.703 |
| MUBARAK | 0.453 | 10/10 | 0.000 | 3/5 | 8.412 |
| NIGHT | 0.138 | 8/12 | 2.750 | 0/0 | 1.052 |
| NMR | 0.308 | 3/4 | 0.083 | 0/0 | 5.938 |
| NOM | 0.200 | 4/7 | 0.000 | 0/1 | 5.430 |
| ORCA | 0.322 | 8/13 | 2.000 | 1/1 | 9.057 |
| PHA | 0.548 | 2/2 | 0.042 | 0/1 | 9.661 |
| QNT | 0.598 | 4/4 | 0.417 | 0/1 | 6.429 |
| RLC | 0.452 | 2/2 | 0.250 | 0/0 | 5.660 |
| SOON | 0.366 | 3/5 | 2.167 | 1/2 | 2.992 |
| TRUMP | 0.500 | 2/2 | 0.375 | 0/0 | 7.500 |
| VELVET | 0.750 | 1/1 | 0.000 | 0/0 | 25.263 |
| ZEC | 0.428 | 7/7 | 0.000 | 3/4 | 23.419 |
| ZRO | 0.227 | 5/5 | 2.750 | 0/0 | 13.019 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.296 | 2/2 | 0.000 | 0/0 | 3.217 |
| AIN | 0.375 | 6/7 | 0.167 | 0/0 | 0.495 |
| AKE | 0.533 | 8/10 | 0.083 | 2/3 | 5.150 |
| API3 | 0.478 | 3/6 | 0.000 | 1/4 | 0.000 |
| ARK | 0.430 | 5/6 | 0.750 | 0/2 | 4.642 |
| ENA | 0.481 | 4/4 | 1.500 | 0/0 | 0.000 |
| G | 0.693 | 7/7 | 0.000 | 0/1 | 5.610 |
| JTO | 0.333 | 5/5 | 1.000 | 0/0 | 14.151 |
| LSK | 0.434 | 2/2 | 0.000 | 2/2 | 8.416 |
| MAGIC | 0.286 | 3/3 | 2.000 | 0/0 | 4.528 |
| MON | 0.150 | 2/3 | 6.000 | 0/0 | 19.811 |
| MOVR | 0.327 | 1/1 | 0.000 | 0/0 | 14.227 |
| MUBARAK | 0.447 | 10/10 | 0.000 | 3/5 | 9.897 |
| NIGHT | 0.444 | 9/12 | 2.500 | 0/0 | 6.837 |
| NMR | 0.200 | 3/4 | 0.083 | 0/0 | 12.371 |
| NOM | 0.167 | 3/7 | 0.000 | 0/1 | 3.801 |
| ORCA | 0.304 | 9/13 | 2.000 | 1/1 | 13.585 |
| PHA | 0.463 | 2/2 | 0.000 | 1/1 | 11.186 |
| QNT | 0.581 | 4/4 | 0.417 | 0/1 | 8.407 |
| RLC | 0.708 | 2/2 | 0.125 | 0/0 | 2.264 |
| SOON | 0.433 | 4/5 | 2.500 | 1/2 | 3.989 |
| TRUMP | 0.545 | 2/2 | 2.250 | 0/0 | 3.500 |
| VELVET | 0.545 | 1/1 | 0.750 | 0/0 | 12.632 |
| ZEC | 0.541 | 7/7 | 0.500 | 3/4 | 10.464 |
| ZRO | 0.286 | 5/5 | 2.250 | 0/0 | 24.906 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.558 | 25.356 |
| oi_z_24h | 0.559 | 16.994 |
| volume_z_7d | 0.719 | 14.563 |
| ls_long_pct | -0.209 | 10.284 |
| ret_4h | 0.287 | 5.895 |
| oi_chg_24h | 0.187 | 4.896 |
| volume_ratio_24h | -0.019 | 3.062 |
| oi_chg_1h | 0.174 | 3.024 |
| funding_z_7d | 0.051 | 2.815 |
| oi_z_7d | 0.271 | 2.205 |
| oi_chg_4h | -0.031 | 1.708 |
| funding_chg_24h | -0.189 | 1.664 |
| funding | 0.066 | 1.563 |
| ret_1h | 0.146 | 1.446 |
| volatility_24h | -0.287 | 1.427 |
| buy_ratio_4h | 0.271 | 1.395 |
| oi_minus_ret_24h | -0.226 | 0.935 |
| buy_ratio_1h | 0.068 | 0.386 |
| range_1h | -0.174 | 0.214 |
| ls_chg_24h | 0.034 | 0.167 |

### LightGBM gain by feature group

Correlated features share gain and can swap places between runs; the group total is more stable.

| group | gain_% | top feature |
|---|---|---|
| Ціна | 32.700 | ret_24h (25.4%) |
| OI | 29.800 | oi_z_24h (17.0%) |
| Обсяг | 17.600 | volume_z_7d (14.6%) |
| Long / Short | 10.500 | ls_long_pct (10.3%) |
| Funding | 6.000 | funding_z_7d (2.8%) |
| Купівлі / продажі | 1.800 | buy_ratio_4h (1.4%) |
| Волатильність | 1.600 | volatility_24h (1.4%) |

## 15m model and signal tiers

LightGBM on 118391 15-minute bars (41 features: the hourly set with the same wall-clock windows plus short bar features and own-move features vs the ALT-EW index — present on 100% of bars), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 2.24 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.345 | 0.504 | 0.409 | 0.364 | 117/130 | 0.833 | 19/27 | 11.336 |
| 15m + 1h ≥ 0.75 (alert) | 0.363 | 0.472 | 0.410 | — | 112/130 | 1.250 | 14/27 | 8.660 |

### 15m LightGBM gain by feature group

| group | gain_% | top feature |
|---|---|---|
| Ціна | 34.000 | ret_24h_vol (26.4%) |
| OI | 21.500 | oi_z_24h (14.9%) |
| Власний рух | 20.400 | idio_24h (7.8%) |
| Long / Short | 7.400 | ls_long_pct (7.4%) |
| Обсяг | 6.500 | volume_z_7d (4.9%) |
| Funding | 5.700 | funding_z_7d (3.7%) |
| Волатильність | 3.400 | park_x (2.0%) |
| Ліквідність (Amihud) | 0.800 | illiq_x (0.8%) |
| Купівлі / продажі | 0.400 | buy_ratio_4h (0.3%) |
