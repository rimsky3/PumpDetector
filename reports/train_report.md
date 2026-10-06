# Training report

Generated 2026-10-05 11:33 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 22557 hourly bars, 21 tickers (AGT, AIN, AKE, ARK, G, JTO, LSK, MAGIC, MON, MOVR, MUBARAK, NIGHT, NMR, NOM, PHA, QNT, SOON, TRUMP, VELVET, ZEC, ZRO), 670 pump hours (3.0%), 101 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.313 | 0.409 | 0.354 | — | 60/100 | 0.500 | 4/22 | 3.160 |
| logreg | 0.260 | 0.454 | 0.331 | 0.314 | 76/100 | 0.042 | 12/22 | 9.288 |
| lightgbm | 0.387 | 0.549 | 0.454 | 0.410 | 78/100 | 0.375 | 10/22 | 6.671 |

## Training settings

- `--fresh-hours 48`: 734 unlabeled hours after 2026-10-03 11:30 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.129 | 1/2 | 0.000 | 0/0 | 2.681 |
| AIN | 0.364 | 4/4 | 0.000 | 0/0 | 1.043 |
| AKE | 0.505 | 6/10 | 0.375 | 1/3 | 0.983 |
| ARK | 0.250 | 5/6 | 3.250 | 0/2 | 6.964 |
| G | 0.338 | 3/7 | 0.583 | 0/1 | 1.966 |
| JTO | 0.000 | 0/5 | — | 0/0 | 0.000 |
| LSK | 0.427 | 2/2 | 0.000 | 2/2 | 2.949 |
| MAGIC | 0.000 | 0/3 | — | 0/0 | 0.000 |
| MON | 0.133 | 1/3 | 15.250 | 0/0 | 12.743 |
| MOVR | 0.312 | 1/1 | 0.500 | 0/0 | 8.036 |
| MUBARAK | 0.429 | 9/10 | 0.000 | 1/5 | 2.609 |
| NIGHT | 0.344 | 5/12 | 2.750 | 0/0 | 1.608 |
| NMR | 0.172 | 2/2 | 0.000 | 0/0 | 4.174 |
| NOM | 0.071 | 2/7 | 0.000 | 0/1 | 1.629 |
| PHA | 0.194 | 2/2 | 5.792 | 0/1 | 5.361 |
| QNT | 0.444 | 3/4 | 2.000 | 0/1 | 5.735 |
| SOON | 0.495 | 3/5 | 0.000 | 0/2 | 1.050 |
| TRUMP | 0.500 | 2/2 | 1.250 | 0/0 | 1.053 |
| VELVET | 0.545 | 1/1 | 0.000 | 0/0 | 34.286 |
| ZEC | 0.609 | 5/7 | 0.500 | 0/4 | 1.050 |
| ZRO | 0.286 | 3/5 | 2.500 | 0/0 | 0.000 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.121 | 2/2 | 0.000 | 0/0 | 9.650 |
| AIN | 0.125 | 3/4 | 0.583 | 0/0 | 7.820 |
| AKE | 0.375 | 9/10 | 0.000 | 2/3 | 11.795 |
| ARK | 0.365 | 5/6 | 1.250 | 0/2 | 8.036 |
| G | 0.509 | 5/7 | 1.583 | 0/1 | 1.474 |
| JTO | 0.000 | 0/5 | — | 0/0 | 0.000 |
| LSK | 0.383 | 2/2 | 0.000 | 2/2 | 10.812 |
| MAGIC | 0.316 | 2/3 | 3.625 | 0/0 | 0.000 |
| MON | 0.160 | 1/3 | 15.250 | 0/0 | 12.743 |
| MOVR | 0.375 | 1/1 | 0.000 | 0/0 | 12.321 |
| MUBARAK | 0.496 | 10/10 | 0.000 | 3/5 | 4.174 |
| NIGHT | 0.158 | 8/12 | 2.750 | 0/0 | 1.072 |
| NMR | 0.143 | 2/2 | 0.042 | 0/0 | 10.435 |
| NOM | 0.167 | 4/7 | 0.000 | 0/1 | 7.059 |
| PHA | 0.472 | 2/2 | 0.000 | 0/1 | 11.794 |
| QNT | 0.491 | 4/4 | 0.000 | 1/1 | 11.470 |
| SOON | 0.315 | 3/5 | 3.000 | 1/2 | 2.626 |
| TRUMP | 0.456 | 2/2 | 0.375 | 0/0 | 10.534 |
| VELVET | 0.600 | 1/1 | 0.000 | 0/0 | 42.857 |
| ZEC | 0.327 | 7/7 | 0.000 | 3/4 | 27.309 |
| ZRO | 0.235 | 3/5 | 3.250 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AGT | 0.250 | 1/2 | 0.000 | 0/0 | 2.144 |
| AIN | 0.303 | 4/4 | 0.292 | 0/0 | 0.000 |
| AKE | 0.444 | 6/10 | 0.750 | 1/3 | 3.932 |
| ARK | 0.464 | 5/6 | 1.250 | 0/2 | 4.821 |
| G | 0.730 | 7/7 | 0.583 | 0/1 | 1.966 |
| JTO | 0.545 | 2/5 | 2.500 | 0/0 | 0.000 |
| LSK | 0.452 | 2/2 | 0.000 | 2/2 | 6.389 |
| MAGIC | 0.400 | 2/3 | 3.125 | 0/0 | 0.000 |
| MON | 0.343 | 1/3 | 9.250 | 0/0 | 25.487 |
| MOVR | 0.386 | 1/1 | 0.000 | 0/0 | 12.857 |
| MUBARAK | 0.429 | 10/10 | 0.000 | 3/5 | 6.261 |
| NIGHT | 0.504 | 10/12 | 2.750 | 0/0 | 5.361 |
| NMR | 0.138 | 2/2 | 0.042 | 0/0 | 10.435 |
| NOM | 0.129 | 2/7 | 0.000 | 0/1 | 3.258 |
| PHA | 0.506 | 2/2 | 0.042 | 0/1 | 9.650 |
| QNT | 0.444 | 4/4 | 0.000 | 0/1 | 13.555 |
| SOON | 0.525 | 3/5 | 2.167 | 1/2 | 2.626 |
| TRUMP | 0.462 | 2/2 | 2.250 | 0/0 | 5.794 |
| VELVET | 0.500 | 1/1 | 3.750 | 0/0 | 0.000 |
| ZEC | 0.450 | 7/7 | 0.500 | 3/4 | 16.280 |
| ZRO | 0.640 | 4/5 | 2.375 | 0/0 | 25.487 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| oi_z_24h | 0.646 | 18.919 |
| volume_z_7d | 0.971 | 14.915 |
| ret_24h | 0.535 | 14.579 |
| oi_chg_24h | 0.270 | 13.308 |
| ls_long_pct | -0.244 | 9.257 |
| volume_ratio_24h | -0.011 | 3.689 |
| oi_chg_4h | 0.219 | 3.472 |
| ret_4h | -0.015 | 3.247 |
| funding | -0.030 | 2.868 |
| oi_chg_1h | -0.011 | 2.636 |
| liq_short_4h_oi | -0.146 | 2.114 |
| funding_z_7d | 0.029 | 2.109 |
| liq_long_4h_oi | -0.753 | 1.473 |
| funding_chg_24h | -0.175 | 1.276 |
| ret_1h | 0.299 | 1.152 |
| oi_minus_ret_24h | -0.117 | 1.072 |
| buy_ratio_4h | 0.237 | 0.888 |
| liq_imbalance_4h | 0.139 | 0.886 |
| volatility_24h | -0.057 | 0.793 |
| oi_z_7d | 0.147 | 0.536 |
| range_1h | -0.285 | 0.381 |
| ls_chg_24h | 0.035 | 0.257 |
| buy_ratio_1h | 0.040 | 0.172 |

## 15m model and signal tiers

LightGBM on 79730 15-minute bars (34 features: the hourly set with the same wall-clock windows plus short bar features), leave-ticker-out with nested threshold. *Candidate* = 15m model alone; *alert* = 15m signal confirmed by the hourly LightGBM at ≥ 0.75 of its threshold (score of the last closed hour). Event delays are in hours from the labeled start; stop multiplier on 15m bars: k = 2.13 ATR.

| tier | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| 15m candidate | 0.391 | 0.511 | 0.443 | 0.394 | 84/100 | 0.583 | 15/22 | 10.728 |
| 15m + 1h ≥ 0.75 (alert) | 0.419 | 0.482 | 0.448 | — | 82/100 | 0.833 | 13/22 | 7.586 |
