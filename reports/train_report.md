# Training report

Generated 2026-09-29 00:43 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 10518 hourly bars, 8 tickers (AIN, AKE, G, LSK, MUBARAK, MYX, NMR, QNT), 392 pump hours (3.7%), 18 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.466 | 0.707 | 0.561 | — | 17/18 | 0.000 | 6/8 | 3.973 |
| logreg | 0.683 | 0.753 | 0.716 | 0.704 | 17/18 | 0.000 | 5/8 | 1.850 |
| lightgbm | 0.668 | 0.806 | 0.731 | 0.741 | 17/18 | 0.000 | 4/8 | 1.233 |

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.780 | 1/1 | 0.000 | 0/0 | 3.962 |
| AKE | 0.697 | 5/6 | 4.000 | 0/1 | 1.062 |
| G | 0.290 | 1/1 | 0.000 | 1/2 | 3.717 |
| LSK | 0.500 | 1/1 | 0.000 | 2/2 | 3.717 |
| MUBARAK | 0.769 | 5/5 | 0.000 | 1/1 | 4.532 |
| MYX | 0.062 | 1/1 | 0.000 | 2/2 | 3.717 |
| NMR | 0.474 | 2/2 | 0.000 | 0/0 | 7.364 |
| QNT | 0.312 | 1/1 | 0.000 | 0/0 | 3.962 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.900 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.739 | 5/6 | 0.000 | 0/1 | 5.310 |
| G | 0.462 | 1/1 | 0.000 | 1/2 | 1.593 |
| LSK | 0.595 | 1/1 | 0.000 | 2/2 | 2.124 |
| MUBARAK | 0.795 | 5/5 | 1.000 | 1/1 | 0.000 |
| MYX | 0.667 | 1/1 | 0.000 | 1/2 | 0.000 |
| NMR | 0.833 | 2/2 | 0.500 | 0/0 | 0.566 |
| QNT | 0.619 | 1/1 | 0.000 | 0/0 | 5.094 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.977 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.777 | 5/6 | 3.000 | 0/1 | 0.000 |
| G | 0.310 | 1/1 | 0.000 | 1/2 | 2.124 |
| LSK | 0.714 | 1/1 | 0.000 | 2/2 | 2.655 |
| MUBARAK | 0.779 | 5/5 | 0.000 | 1/1 | 3.399 |
| MYX | 0.333 | 1/1 | 0.000 | 0/2 | 0.531 |
| NMR | 0.643 | 2/2 | 2.000 | 0/0 | 0.000 |
| QNT | 0.661 | 1/1 | 0.000 | 0/0 | 1.132 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.800 | 51.127 |
| liq_short_4h_oi | -0.409 | 17.857 |
| oi_z_7d | 1.602 | 7.605 |
| oi_chg_24h | 0.642 | 7.292 |
| ls_long_pct | -2.715 | 4.062 |
| volatility_24h | -1.224 | 2.158 |
| volume_z_7d | 0.948 | 1.471 |
| oi_minus_ret_24h | 0.056 | 1.174 |
| volume_ratio_24h | 0.202 | 1.007 |
| oi_chg_4h | -0.599 | 0.809 |
| oi_z_24h | -0.155 | 0.798 |
| oi_chg_1h | -0.101 | 0.794 |
| buy_ratio_4h | 0.447 | 0.669 |
| funding_z_7d | 0.009 | 0.552 |
| liq_imbalance_4h | 0.428 | 0.442 |
| funding | 0.040 | 0.407 |
| ls_chg_24h | 0.279 | 0.402 |
| funding_chg_24h | -0.070 | 0.346 |
| liq_long_4h_oi | 0.013 | 0.319 |
| pred_funding_gap | -0.075 | 0.216 |
| range_1h | 0.125 | 0.188 |
| ret_1h | 0.170 | 0.137 |
| ret_4h | 0.881 | 0.121 |
| buy_ratio_1h | -0.335 | 0.044 |
