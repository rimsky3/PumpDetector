# Training report

Generated 2026-09-30 12:12 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 12960 hourly bars, 10 tickers (AIN, AKE, G, LSK, MUBARAK, MYX, NMR, QNT, SOON, TRUMP), 461 pump hours (3.6%), 21 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.457 | 0.642 | 0.534 | — | 20/21 | 0.000 | 6/8 | 3.889 |
| logreg | 0.633 | 0.714 | 0.671 | 0.683 | 21/21 | 0.000 | 5/8 | 2.611 |
| lightgbm | 0.604 | 0.755 | 0.671 | 0.703 | 19/21 | 0.000 | 5/8 | 1.444 |

## Training settings

- `--fresh-hours 48`: 440 unlabeled hours after 2026-09-28 12:12 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: no

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.780 | 1/1 | 0.000 | 0/0 | 3.994 |
| AKE | 0.652 | 5/6 | 4.000 | 0/1 | 1.070 |
| G | 0.290 | 1/1 | 0.000 | 1/2 | 3.744 |
| LSK | 0.432 | 1/1 | 0.000 | 2/2 | 2.675 |
| MUBARAK | 0.769 | 5/5 | 0.000 | 1/1 | 4.568 |
| MYX | 0.062 | 1/1 | 0.000 | 2/2 | 3.744 |
| NMR | 0.480 | 2/2 | 0.000 | 0/0 | 7.364 |
| QNT | 0.312 | 1/1 | 0.000 | 0/0 | 3.994 |
| SOON | 0.519 | 1/1 | 0.000 | 0/0 | 2.830 |
| TRUMP | 0.414 | 2/2 | 0.500 | 0/0 | 5.192 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.900 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.630 | 6/6 | 0.500 | 0/1 | 14.978 |
| G | 0.409 | 1/1 | 0.000 | 1/2 | 2.140 |
| LSK | 0.447 | 1/1 | 1.000 | 2/2 | 2.140 |
| MUBARAK | 0.845 | 5/5 | 0.000 | 1/1 | 0.571 |
| MYX | 0.444 | 1/1 | 0.000 | 1/2 | 0.535 |
| NMR | 0.895 | 2/2 | 0.500 | 0/0 | 0.566 |
| QNT | 0.574 | 1/1 | 0.000 | 0/0 | 4.564 |
| SOON | 0.600 | 1/1 | 0.000 | 0/0 | 0.000 |
| TRUMP | 0.762 | 2/2 | 1.500 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.965 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.764 | 5/6 | 3.000 | 0/1 | 0.535 |
| G | 0.295 | 1/1 | 0.000 | 1/2 | 1.605 |
| LSK | 0.672 | 1/1 | 0.000 | 2/2 | 4.814 |
| MUBARAK | 0.650 | 5/5 | 0.000 | 1/1 | 6.281 |
| MYX | 0.400 | 1/1 | 0.000 | 1/2 | 0.535 |
| NMR | 0.643 | 2/2 | 2.000 | 0/0 | 0.000 |
| QNT | 0.622 | 1/1 | 0.000 | 0/0 | 0.000 |
| SOON | 0.667 | 1/1 | 1.000 | 0/0 | 0.000 |
| TRUMP | 0.235 | 1/2 | 2.000 | 0/0 | 0.577 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.318 | 39.030 |
| liq_short_4h_oi | -0.194 | 18.877 |
| oi_z_7d | 1.890 | 15.591 |
| volatility_24h | -0.262 | 5.884 |
| volume_ratio_24h | 0.272 | 4.805 |
| ls_long_pct | -2.197 | 4.111 |
| oi_chg_24h | 0.366 | 2.168 |
| volume_z_7d | 1.025 | 1.852 |
| funding | 0.513 | 1.252 |
| oi_z_24h | -0.552 | 1.176 |
| oi_chg_4h | -0.379 | 0.922 |
| ls_chg_24h | 0.749 | 0.789 |
| buy_ratio_4h | 0.427 | 0.730 |
| oi_chg_1h | -0.073 | 0.602 |
| oi_minus_ret_24h | 0.200 | 0.428 |
| funding_z_7d | 0.011 | 0.415 |
| liq_long_4h_oi | -0.176 | 0.354 |
| funding_chg_24h | -0.526 | 0.295 |
| liq_imbalance_4h | 0.321 | 0.250 |
| ret_4h | 0.478 | 0.170 |
| range_1h | 0.208 | 0.123 |
| ret_1h | 0.266 | 0.109 |
| buy_ratio_1h | -0.267 | 0.068 |
