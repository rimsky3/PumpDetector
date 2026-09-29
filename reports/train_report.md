# Training report

Generated 2026-09-29 08:53 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 13124 hourly bars, 10 tickers (AIN, AKE, G, LSK, MUBARAK, MYX, NMR, QNT, SOON, ZEC), 872 pump hours (6.6%), 22 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.360 | 0.502 | 0.420 | — | 21/22 | 0.000 | 7/8 | 6.149 |
| logreg | 0.637 | 0.585 | 0.610 | 0.705 | 21/22 | 0.000 | 5/8 | 2.800 |
| lightgbm | 0.553 | 0.569 | 0.561 | 0.570 | 21/22 | 0.000 | 4/8 | 2.526 |

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.704 | 1/1 | 0.000 | 0/0 | 9.000 |
| AKE | 0.759 | 5/6 | 3.000 | 0/1 | 0.528 |
| G | 0.149 | 1/1 | 0.000 | 2/2 | 13.196 |
| LSK | 0.419 | 1/1 | 0.000 | 2/2 | 4.223 |
| MUBARAK | 0.705 | 5/5 | 0.000 | 1/1 | 1.689 |
| MYX | 0.037 | 1/1 | 0.000 | 2/2 | 4.751 |
| NMR | 0.245 | 2/2 | 0.000 | 0/0 | 8.444 |
| QNT | 0.422 | 1/1 | 0.000 | 0/0 | 11.250 |
| SOON | 0.523 | 1/1 | 0.000 | 0/0 | 8.504 |
| ZEC | 0.203 | 3/3 | 5.000 | 0/0 | 0.000 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 1.000 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.634 | 5/6 | 0.000 | 0/1 | 5.806 |
| G | 0.375 | 1/1 | 0.000 | 1/2 | 2.639 |
| LSK | 0.515 | 1/1 | 0.000 | 2/2 | 5.279 |
| MUBARAK | 0.746 | 5/5 | 0.000 | 1/1 | 6.192 |
| MYX | 0.444 | 1/1 | 0.000 | 1/2 | 0.000 |
| NMR | 0.680 | 2/2 | 0.500 | 0/0 | 1.126 |
| QNT | 0.557 | 1/1 | 0.000 | 0/0 | 5.062 |
| SOON | 0.585 | 1/1 | 0.000 | 0/0 | 1.134 |
| ZEC | 0.538 | 3/3 | 0.000 | 0/0 | 0.567 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.989 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.625 | 5/6 | 0.000 | 0/1 | 3.167 |
| G | 0.281 | 1/1 | 0.000 | 1/2 | 1.056 |
| LSK | 0.693 | 1/1 | 0.000 | 2/2 | 3.695 |
| MUBARAK | 0.579 | 5/5 | 0.000 | 1/1 | 7.881 |
| MYX | 0.444 | 1/1 | 0.000 | 0/2 | 0.528 |
| NMR | 0.412 | 2/2 | 2.000 | 0/0 | 0.000 |
| QNT | 0.593 | 1/1 | 0.000 | 0/0 | 9.000 |
| SOON | 0.659 | 1/1 | 1.000 | 0/0 | 0.000 |
| ZEC | 0.429 | 3/3 | 0.000 | 0/0 | 0.000 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| oi_z_7d | 2.563 | 47.100 |
| ls_long_pct | -3.335 | 15.950 |
| liq_short_4h_oi | -0.168 | 15.671 |
| volume_ratio_24h | 0.412 | 5.155 |
| oi_chg_24h | 0.286 | 3.908 |
| volatility_24h | -0.757 | 3.286 |
| liq_long_4h_oi | -0.262 | 2.002 |
| volume_z_7d | 0.609 | 0.971 |
| oi_minus_ret_24h | 0.089 | 0.938 |
| funding | 0.360 | 0.931 |
| ret_24h | 0.308 | 0.793 |
| liq_imbalance_4h | 0.020 | 0.544 |
| ls_chg_24h | 0.789 | 0.522 |
| funding_z_7d | 0.006 | 0.491 |
| oi_z_24h | -0.332 | 0.314 |
| oi_chg_4h | -0.472 | 0.263 |
| ret_4h | 0.577 | 0.242 |
| oi_chg_1h | -0.092 | 0.209 |
| range_1h | 0.097 | 0.206 |
| funding_chg_24h | -0.396 | 0.204 |
| pred_funding_gap | -0.018 | 0.181 |
| buy_ratio_4h | 0.193 | 0.066 |
| ret_1h | 0.163 | 0.053 |
| buy_ratio_1h | -0.024 | 0.001 |
