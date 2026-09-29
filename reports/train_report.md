# Training report

Generated 2026-09-29 00:11 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 10510 hourly bars, 8 tickers (AIN, AKE, G, LSK, MUBARAK, MYX, NMR, QNT), 380 pump hours (3.6%), 17 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.455 | 0.711 | 0.554 | — | 16/17 | 0.000 | 7/9 | 3.976 |
| logreg | 0.657 | 0.739 | 0.696 | 0.671 | 16/17 | 0.000 | 5/9 | 1.645 |
| lightgbm | 0.655 | 0.803 | 0.721 | 0.728 | 16/17 | 0.000 | 5/9 | 1.371 |

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.780 | 1/1 | 0.000 | 0/0 | 3.965 |
| AKE | 0.667 | 4/5 | 3.000 | 1/2 | 1.063 |
| G | 0.290 | 1/1 | 0.000 | 1/2 | 3.720 |
| LSK | 0.500 | 1/1 | 0.000 | 2/2 | 3.720 |
| MUBARAK | 0.769 | 5/5 | 0.000 | 1/1 | 4.535 |
| MYX | 0.062 | 1/1 | 0.000 | 2/2 | 3.720 |
| NMR | 0.480 | 2/2 | 0.000 | 0/0 | 7.370 |
| QNT | 0.312 | 1/1 | 0.000 | 0/0 | 3.965 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.914 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.678 | 4/5 | 2.000 | 1/2 | 5.314 |
| G | 0.462 | 1/1 | 0.000 | 1/2 | 1.594 |
| LSK | 0.571 | 1/1 | 0.000 | 2/2 | 2.125 |
| MUBARAK | 0.781 | 5/5 | 1.000 | 0/1 | 0.000 |
| MYX | 0.667 | 1/1 | 0.000 | 1/2 | 0.000 |
| NMR | 0.857 | 2/2 | 0.500 | 0/0 | 0.567 |
| QNT | 0.636 | 1/1 | 0.000 | 0/0 | 3.399 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.927 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.729 | 4/5 | 2.000 | 1/2 | 0.000 |
| G | 0.346 | 1/1 | 0.000 | 1/2 | 2.657 |
| LSK | 0.702 | 1/1 | 0.000 | 2/2 | 2.657 |
| MUBARAK | 0.757 | 5/5 | 0.000 | 1/1 | 3.402 |
| MYX | 0.267 | 1/1 | 0.000 | 0/2 | 1.594 |
| NMR | 0.774 | 2/2 | 2.000 | 0/0 | 0.000 |
| QNT | 0.727 | 1/1 | 0.000 | 0/0 | 0.566 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.791 | 46.946 |
| liq_short_4h_oi | -0.388 | 21.672 |
| oi_chg_24h | 0.635 | 8.331 |
| oi_z_7d | 1.556 | 7.894 |
| ls_long_pct | -2.589 | 3.802 |
| volatility_24h | -1.214 | 1.824 |
| oi_minus_ret_24h | 0.053 | 1.122 |
| volume_z_7d | 0.900 | 1.119 |
| oi_z_24h | -0.152 | 0.921 |
| volume_ratio_24h | 0.207 | 0.907 |
| oi_chg_1h | -0.074 | 0.858 |
| funding | 0.026 | 0.618 |
| ls_chg_24h | 0.242 | 0.616 |
| buy_ratio_4h | 0.425 | 0.569 |
| oi_chg_4h | -0.570 | 0.513 |
| funding_z_7d | 0.010 | 0.507 |
| liq_long_4h_oi | 0.022 | 0.309 |
| funding_chg_24h | -0.061 | 0.309 |
| liq_imbalance_4h | 0.430 | 0.264 |
| pred_funding_gap | -0.082 | 0.229 |
| range_1h | 0.139 | 0.228 |
| ret_4h | 0.826 | 0.221 |
| ret_1h | 0.141 | 0.130 |
| buy_ratio_1h | -0.326 | 0.092 |
