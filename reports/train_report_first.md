# Training report

Generated 2026-10-12 14:31 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 34415 hourly bars, 31 tickers (AERO, AGT, AIN, AKE, API3, ARK, BAT, CHIP, CKB, ENA, G, JTO, LSK, MAGIC, MON, MOVR, MUBARAK, NIGHT, NMR, NOM, ORCA, PHA, QNT, RLC, S, SOON, STRK, TRUMP, VELVET, ZEC, ZRO), 978 pump hours (2.8%), 154 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.272 | 0.369 | 0.314 | — | 89/152 | 0.500 | 5/25 | 3.452 |
| logreg | 0.276 | 0.411 | 0.330 | 0.283 | 120/152 | 0.542 | 14/25 | 8.933 |
| lightgbm | 0.328 | 0.511 | 0.400 | 0.315 | 127/152 | 0.750 | 14/25 | 9.666 |

## Training settings

- `--fresh-hours 48`: 253 unlabeled hours after 2026-10-10 14:25 UTC left out of training
- models: rule, logreg, lightgbm; 20 features (without liquidations); saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AERO | 0.000 | 0/4 | — | 0/0 | 0.000 |
| AGT | 0.129 | 1/2 | 0.000 | 0/0 | 2.681 |
| AIN | 0.368 | 6/7 | 0.000 | 0/0 | 0.928 |
| AKE | 0.505 | 6/10 | 0.375 | 1/3 | 0.905 |
| API3 | 0.696 | 2/4 | 0.000 | 0/1 | 0.000 |
| ARK | 0.229 | 5/6 | 3.250 | 0/2 | 6.189 |
| BAT | 0.182 | 1/3 | 1.250 | 0/0 | 7.660 |
| CHIP | 0.290 | 4/6 | 3.500 | 0/0 | 3.475 |
| CKB | 0.455 | 4/4 | 1.125 | 0/0 | 18.462 |
| ENA | 0.423 | 2/4 | 0.750 | 0/0 | 0.000 |
| G | 0.338 | 3/7 | 0.583 | 0/1 | 1.862 |
| JTO | 0.000 | 0/5 | — | 0/0 | 1.698 |
| LSK | 0.427 | 2/2 | 0.000 | 2/2 | 2.793 |
| MAGIC | 0.027 | 1/3 | 0.500 | 0/0 | 2.746 |
| MON | 0.090 | 2/3 | 9.500 | 0/0 | 6.792 |
| MOVR | 0.312 | 1/1 | 0.500 | 0/0 | 7.392 |
| MUBARAK | 0.390 | 9/10 | 0.000 | 1/5 | 3.787 |
| NIGHT | 0.344 | 5/12 | 2.750 | 0/0 | 1.578 |
| NMR | 0.227 | 3/4 | 0.000 | 0/0 | 4.300 |
| NOM | 0.071 | 2/7 | 0.000 | 0/1 | 1.629 |
| ORCA | 0.214 | 6/13 | 0.750 | 1/1 | 3.310 |
| PHA | 0.194 | 2/2 | 5.792 | 0/1 | 5.049 |
| QNT | 0.400 | 3/4 | 2.000 | 0/1 | 5.414 |
| RLC | 0.590 | 2/2 | 0.125 | 0/0 | 4.394 |
| S | 0.250 | 1/4 | 9.750 | 0/0 | 0.000 |
| SOON | 0.495 | 3/5 | 0.000 | 0/2 | 0.990 |
| STRK | 0.034 | 1/3 | 19.750 | 0/1 | 6.000 |
| TRUMP | 0.500 | 2/2 | 1.250 | 0/0 | 0.963 |
| VELVET | 0.500 | 1/1 | 0.000 | 0/0 | 3.475 |
| ZEC | 0.609 | 5/7 | 0.500 | 0/4 | 0.936 |
| ZRO | 0.157 | 4/5 | 3.125 | 0/0 | 10.189 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AERO | 0.231 | 2/4 | 6.125 | 0/0 | 0.000 |
| AGT | 0.170 | 2/2 | 0.000 | 0/0 | 7.506 |
| AIN | 0.293 | 5/7 | 0.000 | 0/0 | 6.495 |
| AKE | 0.400 | 9/10 | 0.000 | 2/3 | 12.219 |
| API3 | 0.636 | 2/4 | 0.375 | 0/1 | 0.000 |
| ARK | 0.311 | 5/6 | 1.250 | 0/2 | 7.221 |
| BAT | 0.286 | 2/3 | 2.125 | 0/0 | 30.638 |
| CHIP | 0.264 | 5/6 | 1.250 | 0/0 | 12.164 |
| CKB | 0.364 | 3/4 | 2.250 | 0/0 | 18.462 |
| ENA | 0.327 | 4/4 | 1.125 | 0/0 | 0.000 |
| G | 0.535 | 7/7 | 0.500 | 0/1 | 4.654 |
| JTO | 0.178 | 3/5 | 2.000 | 0/0 | 7.358 |
| LSK | 0.327 | 2/2 | 0.000 | 2/2 | 12.566 |
| MAGIC | 0.149 | 3/3 | 2.250 | 0/0 | 7.689 |
| MON | 0.160 | 2/3 | 9.000 | 0/0 | 7.925 |
| MOVR | 0.372 | 1/1 | 0.000 | 0/0 | 12.320 |
| MUBARAK | 0.428 | 10/10 | 0.000 | 3/5 | 8.047 |
| NIGHT | 0.137 | 8/12 | 2.750 | 0/0 | 1.052 |
| NMR | 0.209 | 3/4 | 0.083 | 0/0 | 11.466 |
| NOM | 0.208 | 4/7 | 0.000 | 0/1 | 5.430 |
| ORCA | 0.286 | 7/13 | 2.000 | 1/1 | 8.828 |
| PHA | 0.548 | 2/2 | 0.042 | 0/1 | 9.593 |
| QNT | 0.479 | 4/4 | 0.000 | 1/1 | 11.319 |
| RLC | 0.438 | 2/2 | 0.250 | 0/0 | 8.787 |
| S | 0.286 | 1/4 | 7.750 | 0/0 | 23.607 |
| SOON | 0.417 | 4/5 | 2.083 | 1/2 | 2.971 |
| STRK | 0.179 | 3/3 | 2.250 | 1/1 | 15.000 |
| TRUMP | 0.500 | 2/2 | 0.375 | 0/0 | 6.261 |
| VELVET | 0.480 | 1/1 | 0.000 | 0/0 | 6.372 |
| ZEC | 0.419 | 7/7 | 0.250 | 3/4 | 21.988 |
| ZRO | 0.220 | 5/5 | 2.750 | 0/0 | 9.623 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AERO | 0.414 | 3/4 | 4.250 | 0/0 | 0.000 |
| AGT | 0.240 | 2/2 | 0.375 | 0/0 | 3.217 |
| AIN | 0.375 | 6/7 | 0.167 | 0/0 | 1.392 |
| AKE | 0.500 | 7/10 | 0.167 | 2/3 | 5.883 |
| API3 | 0.696 | 2/4 | 0.000 | 0/1 | 0.000 |
| ARK | 0.440 | 6/6 | 1.000 | 0/2 | 6.189 |
| BAT | 0.455 | 2/3 | 1.625 | 0/0 | 30.638 |
| CHIP | 0.325 | 5/6 | 0.750 | 0/0 | 17.957 |
| CKB | 0.455 | 4/4 | 1.125 | 0/0 | 18.462 |
| ENA | 0.561 | 4/4 | 1.625 | 0/0 | 0.000 |
| G | 0.658 | 7/7 | 0.000 | 0/1 | 4.654 |
| JTO | 0.257 | 5/5 | 1.000 | 0/0 | 18.679 |
| LSK | 0.463 | 2/2 | 0.000 | 2/2 | 7.912 |
| MAGIC | 0.216 | 3/3 | 2.000 | 0/0 | 6.041 |
| MON | 0.140 | 2/3 | 6.000 | 0/0 | 21.509 |
| MOVR | 0.340 | 1/1 | 0.000 | 0/0 | 14.784 |
| MUBARAK | 0.439 | 10/10 | 0.000 | 3/5 | 9.941 |
| NIGHT | 0.426 | 10/12 | 2.750 | 0/0 | 3.682 |
| NMR | 0.222 | 3/4 | 0.083 | 0/0 | 13.378 |
| NOM | 0.171 | 3/7 | 0.000 | 0/1 | 3.801 |
| ORCA | 0.290 | 8/13 | 2.000 | 1/1 | 13.793 |
| PHA | 0.472 | 2/2 | 0.000 | 1/1 | 12.623 |
| QNT | 0.628 | 4/4 | 0.417 | 0/1 | 8.859 |
| RLC | 0.638 | 2/2 | 0.250 | 0/0 | 4.943 |
| S | 0.667 | 2/4 | 2.750 | 0/0 | 0.000 |
| SOON | 0.533 | 4/5 | 2.083 | 1/2 | 3.961 |
| STRK | 0.175 | 3/3 | 2.250 | 1/1 | 27.000 |
| TRUMP | 0.474 | 2/2 | 2.250 | 0/0 | 4.334 |
| VELVET | 0.500 | 1/1 | 0.750 | 0/0 | 1.738 |
| ZEC | 0.515 | 7/7 | 0.250 | 3/4 | 13.567 |
| ZRO | 0.295 | 5/5 | 0.500 | 0/0 | 30.566 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.530 | 24.819 |
| volume_z_7d | 0.737 | 12.832 |
| oi_z_24h | 0.506 | 12.801 |
| ls_long_pct | -0.185 | 11.580 |
| oi_chg_24h | 0.168 | 5.956 |
| ret_4h | 0.232 | 4.820 |
| volume_ratio_24h | -0.003 | 3.818 |
| oi_chg_4h | 0.022 | 3.394 |
| funding | 0.075 | 3.275 |
| oi_z_7d | 0.180 | 3.222 |
| oi_chg_1h | 0.162 | 3.189 |
| funding_z_7d | 0.046 | 2.899 |
| funding_chg_24h | -0.171 | 1.730 |
| ret_1h | 0.166 | 1.691 |
| volatility_24h | -0.139 | 1.564 |
| oi_minus_ret_24h | -0.238 | 0.853 |
| buy_ratio_4h | 0.180 | 0.733 |
| ls_chg_24h | 0.040 | 0.357 |
| range_1h | -0.342 | 0.283 |
| buy_ratio_1h | 0.049 | 0.184 |

### LightGBM gain by feature group

Correlated features share gain and can swap places between runs; the group total is more stable.

| group | gain_% | top feature |
|---|---|---|
| Ціна | 31.300 | ret_24h (24.8%) |
| OI | 29.400 | oi_z_24h (12.8%) |
| Обсяг | 16.700 | volume_z_7d (12.8%) |
| Long / Short | 11.900 | ls_long_pct (11.6%) |
| Funding | 7.900 | funding (3.3%) |
| Волатильність | 1.800 | volatility_24h (1.6%) |
| Купівлі / продажі | 0.900 | buy_ratio_4h (0.7%) |

## 15m model and signal tiers

LightGBM on 142556 15-minute bars (41 features: the hourly set with the same wall-clock windows plus short bar features and own-move features vs the ALT-EW index — present on 100% of bars), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 2.00 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.329 | 0.466 | 0.386 | 0.328 | 133/152 | 1.000 | 20/25 | 10.990 |
| 15m + 1h ≥ 0.75 (alert) | 0.345 | 0.439 | 0.386 | — | 128/152 | 1.250 | 17/25 | 8.687 |

### 15m LightGBM gain by feature group

| group | gain_% | top feature |
|---|---|---|
| Власний рух | 37.600 | idio_24h (20.4%) |
| Ціна | 22.400 | ret_24h_vol (19.0%) |
| OI | 16.200 | oi_z_24h (10.1%) |
| Long / Short | 9.200 | ls_long_pct (9.0%) |
| Обсяг | 6.000 | volume_z_7d (3.7%) |
| Funding | 5.300 | funding_z_7d (3.5%) |
| Волатильність | 2.900 | park_x (1.7%) |
| Ліквідність (Amihud) | 0.500 | illiq_x (0.5%) |
| Купівлі / продажі | 0.100 | buy_ratio_4h (0.1%) |

## 15m start model (split mode entry)

Target: bars closing in the first 30 minutes of an impulse (the rest of the impulse is left out of training). Threshold matched to the main 15m model's false-alarm rate (11.0 per 30 days): 0.907. Leave-ticker-out: caught 122/152, false starts 20/25, false alarms 11.9 per 30 days, median delay 0.75 h.
