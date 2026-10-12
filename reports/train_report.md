# Training report

Generated 2026-10-12 15:03 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 39970 hourly bars, 32 tickers (AERO, AGT, AIN, AKE, API3, ARK, BAT, CHIP, CKB, ENA, G, JTO, LSK, MAGIC, MON, MOVR, MUBARAK, NEAR, NIGHT, NMR, NOM, ORCA, PHA, QNT, RLC, S, SOON, STRK, TRUMP, VELVET, ZEC, ZRO), 1006 pump hours (2.5%), 154 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.245 | 0.364 | 0.293 | — | 90/153 | 0.500 | 5/25 | 3.963 |
| logreg | 0.287 | 0.370 | 0.323 | 0.261 | 116/153 | 0.667 | 14/25 | 7.241 |
| lightgbm | 0.298 | 0.464 | 0.363 | 0.278 | 125/153 | 0.500 | 14/25 | 8.827 |

## Training settings

- `--fresh-hours 48`: 343 unlabeled hours after 2026-10-10 14:56 UTC left out of training
- models: rule, logreg, lightgbm; 20 features (without liquidations); saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AERO | 0.048 | 1/4 | 10.750 | 0/0 | 9.412 |
| AGT | 0.129 | 1/2 | 0.000 | 0/0 | 2.681 |
| AIN | 0.368 | 6/7 | 0.000 | 0/0 | 0.928 |
| AKE | 0.505 | 6/10 | 0.375 | 1/3 | 0.905 |
| API3 | 0.696 | 2/4 | 0.000 | 0/1 | 0.000 |
| ARK | 0.250 | 5/6 | 3.250 | 0/2 | 6.705 |
| BAT | 0.066 | 1/3 | 1.250 | 0/0 | 6.471 |
| CHIP | 0.290 | 4/6 | 3.500 | 0/0 | 3.475 |
| CKB | 0.327 | 4/4 | 1.125 | 0/0 | 5.835 |
| ENA | 0.423 | 2/4 | 0.750 | 0/0 | 0.000 |
| G | 0.338 | 3/7 | 0.583 | 0/1 | 1.862 |
| JTO | 0.000 | 0/5 | — | 0/0 | 1.698 |
| LSK | 0.427 | 2/2 | 0.000 | 2/2 | 2.793 |
| MAGIC | 0.027 | 1/3 | 0.500 | 0/0 | 2.746 |
| MON | 0.090 | 2/3 | 9.500 | 0/0 | 6.792 |
| MOVR | 0.312 | 1/1 | 0.500 | 0/0 | 7.392 |
| MUBARAK | 0.390 | 9/10 | 0.000 | 1/5 | 3.787 |
| NEAR | 0.000 | 0/1 | — | 0/0 | 0.000 |
| NIGHT | 0.344 | 5/12 | 2.750 | 0/0 | 1.578 |
| NMR | 0.227 | 3/4 | 0.000 | 0/0 | 4.300 |
| NOM | 0.071 | 2/7 | 0.000 | 0/1 | 1.629 |
| ORCA | 0.214 | 6/13 | 0.750 | 1/1 | 3.310 |
| PHA | 0.194 | 2/2 | 5.792 | 0/1 | 5.049 |
| QNT | 0.444 | 3/4 | 2.000 | 0/1 | 5.414 |
| RLC | 0.590 | 2/2 | 0.125 | 0/0 | 4.394 |
| S | 0.075 | 1/4 | 9.750 | 0/0 | 5.234 |
| SOON | 0.495 | 3/5 | 0.000 | 0/2 | 0.990 |
| STRK | 0.018 | 1/3 | 19.750 | 0/1 | 7.641 |
| TRUMP | 0.500 | 2/2 | 1.250 | 0/0 | 0.963 |
| VELVET | 0.500 | 1/1 | 0.000 | 0/0 | 3.475 |
| ZEC | 0.609 | 5/7 | 0.500 | 0/4 | 0.936 |
| ZRO | 0.157 | 4/5 | 3.125 | 0/0 | 10.189 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AERO | 0.160 | 3/4 | 8.750 | 0/0 | 14.118 |
| AGT | 0.258 | 2/2 | 0.000 | 0/0 | 5.361 |
| AIN | 0.306 | 6/7 | 0.000 | 0/0 | 6.959 |
| AKE | 0.367 | 9/10 | 0.000 | 2/3 | 14.481 |
| API3 | 0.636 | 2/4 | 0.375 | 0/1 | 0.000 |
| ARK | 0.323 | 5/6 | 1.250 | 0/2 | 5.673 |
| BAT | 0.113 | 2/3 | 2.125 | 0/0 | 10.000 |
| CHIP | 0.197 | 4/6 | 0.750 | 0/0 | 8.109 |
| CKB | 0.242 | 3/4 | 2.250 | 0/0 | 4.668 |
| ENA | 0.217 | 4/4 | 2.125 | 0/0 | 0.000 |
| G | 0.525 | 6/7 | 0.292 | 0/1 | 1.862 |
| JTO | 0.071 | 1/5 | 3.000 | 0/0 | 0.566 |
| LSK | 0.414 | 2/2 | 0.000 | 2/2 | 8.843 |
| MAGIC | 0.113 | 2/3 | 2.250 | 0/0 | 6.041 |
| MON | 0.158 | 2/3 | 9.500 | 0/0 | 5.094 |
| MOVR | 0.337 | 1/1 | 0.000 | 0/0 | 11.828 |
| MUBARAK | 0.446 | 10/10 | 0.000 | 3/5 | 7.574 |
| NEAR | 0.667 | 1/1 | 1.000 | 0/0 | 0.000 |
| NIGHT | 0.139 | 8/12 | 2.750 | 0/0 | 1.052 |
| NMR | 0.383 | 3/4 | 0.083 | 0/0 | 4.778 |
| NOM | 0.227 | 4/7 | 0.000 | 0/1 | 4.887 |
| ORCA | 0.277 | 7/13 | 2.000 | 1/1 | 8.828 |
| PHA | 0.588 | 2/2 | 0.042 | 0/1 | 9.593 |
| QNT | 0.542 | 4/4 | 0.417 | 1/1 | 7.874 |
| RLC | 0.538 | 2/2 | 0.250 | 0/0 | 7.689 |
| S | 0.192 | 3/4 | 2.000 | 0/0 | 9.887 |
| SOON | 0.326 | 3/5 | 3.000 | 1/2 | 1.981 |
| STRK | 0.119 | 3/3 | 2.250 | 1/1 | 11.167 |
| TRUMP | 0.625 | 2/2 | 2.250 | 0/0 | 1.926 |
| VELVET | 0.600 | 1/1 | 0.000 | 0/0 | 4.634 |
| ZEC | 0.486 | 7/7 | 0.250 | 3/4 | 15.906 |
| ZRO | 0.100 | 2/5 | 3.125 | 0/0 | 8.491 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AERO | 0.155 | 4/4 | 4.375 | 0/0 | 18.235 |
| AGT | 0.308 | 2/2 | 0.000 | 0/0 | 3.217 |
| AIN | 0.348 | 6/7 | 0.292 | 0/0 | 0.928 |
| AKE | 0.544 | 9/10 | 0.000 | 2/3 | 6.788 |
| API3 | 0.696 | 2/4 | 0.000 | 0/1 | 0.000 |
| ARK | 0.500 | 5/6 | 0.250 | 0/2 | 6.189 |
| BAT | 0.156 | 2/3 | 1.625 | 0/0 | 13.529 |
| CHIP | 0.276 | 5/6 | 1.250 | 0/0 | 18.536 |
| CKB | 0.372 | 4/4 | 1.125 | 0/0 | 7.002 |
| ENA | 0.481 | 4/4 | 1.500 | 0/0 | 0.000 |
| G | 0.687 | 7/7 | 0.000 | 0/1 | 3.723 |
| JTO | 0.226 | 3/5 | 2.000 | 0/0 | 9.057 |
| LSK | 0.460 | 2/2 | 0.000 | 2/2 | 8.378 |
| MAGIC | 0.200 | 3/3 | 2.000 | 0/0 | 4.394 |
| MON | 0.139 | 2/3 | 6.000 | 0/0 | 18.113 |
| MOVR | 0.320 | 1/1 | 0.000 | 0/0 | 14.784 |
| MUBARAK | 0.468 | 10/10 | 0.000 | 3/5 | 8.994 |
| NEAR | 1.000 | 1/1 | 0.000 | 0/0 | 0.000 |
| NIGHT | 0.341 | 8/12 | 2.750 | 0/0 | 4.207 |
| NMR | 0.277 | 3/4 | 0.083 | 0/0 | 10.989 |
| NOM | 0.176 | 3/7 | 0.000 | 0/1 | 3.801 |
| ORCA | 0.306 | 7/13 | 2.000 | 1/1 | 8.276 |
| PHA | 0.500 | 2/2 | 0.042 | 0/1 | 9.088 |
| QNT | 0.614 | 4/4 | 0.417 | 1/1 | 8.859 |
| RLC | 0.682 | 2/2 | 0.250 | 0/0 | 3.295 |
| S | 0.195 | 4/4 | 1.875 | 0/0 | 18.029 |
| SOON | 0.468 | 4/5 | 2.083 | 1/2 | 2.476 |
| STRK | 0.150 | 3/3 | 2.250 | 1/1 | 10.580 |
| TRUMP | 0.414 | 1/2 | 3.750 | 0/0 | 2.890 |
| VELVET | 0.000 | 0/1 | — | 0/0 | 0.579 |
| ZEC | 0.500 | 7/7 | 0.500 | 3/4 | 9.825 |
| ZRO | 0.284 | 5/5 | 2.250 | 0/0 | 27.736 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.509 | 28.021 |
| volume_z_7d | 0.651 | 11.188 |
| ls_long_pct | -0.248 | 10.492 |
| oi_z_24h | 0.501 | 10.376 |
| oi_chg_24h | 0.170 | 6.440 |
| ret_4h | 0.233 | 4.807 |
| oi_chg_4h | 0.013 | 3.889 |
| volume_ratio_24h | -0.006 | 3.672 |
| funding | 0.075 | 3.285 |
| oi_chg_1h | 0.121 | 3.204 |
| oi_z_7d | 0.178 | 2.988 |
| funding_z_7d | 0.058 | 2.465 |
| volatility_24h | -0.088 | 2.321 |
| ret_1h | 0.189 | 2.045 |
| funding_chg_24h | -0.164 | 1.661 |
| oi_minus_ret_24h | -0.215 | 1.094 |
| buy_ratio_4h | 0.161 | 0.793 |
| ls_chg_24h | 0.063 | 0.503 |
| range_1h | -0.273 | 0.439 |
| buy_ratio_1h | 0.070 | 0.318 |

### LightGBM gain by feature group

Correlated features share gain and can swap places between runs; the group total is more stable.

| group | gain_% | top feature |
|---|---|---|
| Ціна | 34.900 | ret_24h (28.0%) |
| OI | 28.000 | oi_z_24h (10.4%) |
| Обсяг | 14.900 | volume_z_7d (11.2%) |
| Long / Short | 11.000 | ls_long_pct (10.5%) |
| Funding | 7.400 | funding (3.3%) |
| Волатильність | 2.800 | volatility_24h (2.3%) |
| Купівлі / продажі | 1.100 | buy_ratio_4h (0.8%) |

## 15m model and signal tiers

LightGBM on 155690 15-minute bars (41 features: the hourly set with the same wall-clock windows plus short bar features and own-move features vs the ALT-EW index — present on 98% of bars), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 1.95 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.321 | 0.451 | 0.375 | 0.308 | 135/153 | 1.000 | 19/25 | 10.359 |
| 15m + 1h ≥ 0.75 (alert) | 0.338 | 0.423 | 0.376 | — | 129/153 | 1.250 | 17/25 | 8.269 |

### 15m LightGBM gain by feature group

| group | gain_% | top feature |
|---|---|---|
| Власний рух | 37.500 | idio_24h (19.8%) |
| Ціна | 20.700 | ret_24h_vol (17.5%) |
| OI | 16.900 | oi_z_24h (10.6%) |
| Long / Short | 9.400 | ls_long_pct (9.2%) |
| Funding | 5.700 | funding_z_7d (3.7%) |
| Обсяг | 5.600 | volume_z_7d (3.0%) |
| Волатильність | 3.500 | park_x (2.5%) |
| Ліквідність (Amihud) | 0.500 | illiq_x (0.5%) |
| Купівлі / продажі | 0.100 | buy_ratio_4h (0.1%) |

## 15m start model (split mode entry)

Target: bars closing in the first 30 minutes of an impulse (the rest of the impulse is left out of training). Threshold matched to the main 15m model's false-alarm rate (10.4 per 30 days): 0.914. Leave-ticker-out: caught 114/153, false starts 22/25, false alarms 9.8 per 30 days, median delay 0.75 h.
