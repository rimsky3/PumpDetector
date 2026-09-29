# Training report

Generated 2026-09-29 09:07 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 11862 hourly bars, 9 tickers (AIN, AKE, G, LSK, MUBARAK, MYX, NMR, QNT, SOON), 448 pump hours (3.8%), 19 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.474 | 0.705 | 0.567 | — | 18/19 | 0.000 | 6/8 | 3.827 |
| logreg | 0.585 | 0.696 | 0.636 | 0.645 | 18/19 | 0.000 | 5/8 | 2.612 |
| lightgbm | 0.652 | 0.779 | 0.710 | 0.716 | 18/19 | 0.000 | 3/8 | 1.640 |

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.780 | 1/1 | 0.000 | 0/0 | 3.934 |
| AKE | 0.697 | 5/6 | 4.000 | 0/1 | 1.055 |
| G | 0.290 | 1/1 | 0.000 | 1/2 | 3.692 |
| LSK | 0.500 | 1/1 | 0.000 | 2/2 | 3.692 |
| MUBARAK | 0.769 | 5/5 | 0.000 | 1/1 | 4.500 |
| MYX | 0.062 | 1/1 | 0.000 | 2/2 | 3.692 |
| NMR | 0.424 | 2/2 | 0.000 | 0/0 | 7.312 |
| QNT | 0.468 | 1/1 | 0.000 | 0/0 | 3.934 |
| SOON | 0.519 | 1/1 | 0.000 | 0/0 | 2.832 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.872 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.583 | 5/6 | 0.000 | 0/1 | 12.132 |
| G | 0.462 | 1/1 | 0.000 | 1/2 | 1.582 |
| LSK | 0.421 | 1/1 | 15.000 | 2/2 | 2.637 |
| MUBARAK | 0.810 | 5/5 | 0.000 | 1/1 | 2.250 |
| MYX | 0.571 | 1/1 | 0.000 | 1/2 | 0.000 |
| NMR | 0.708 | 2/2 | 0.500 | 0/0 | 0.562 |
| QNT | 0.530 | 1/1 | 0.000 | 0/0 | 3.934 |
| SOON | 0.600 | 1/1 | 0.000 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 1.000 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.777 | 5/6 | 3.000 | 0/1 | 0.527 |
| G | 0.240 | 1/1 | 0.000 | 0/2 | 3.692 |
| LSK | 0.828 | 1/1 | 0.000 | 2/2 | 2.110 |
| MUBARAK | 0.695 | 5/5 | 0.000 | 1/1 | 6.750 |
| MYX | 0.500 | 1/1 | 0.000 | 0/2 | 0.527 |
| NMR | 0.432 | 2/2 | 2.000 | 0/0 | 0.000 |
| QNT | 0.646 | 1/1 | 0.000 | 0/0 | 1.124 |
| SOON | 0.651 | 1/1 | 1.000 | 0/0 | 0.000 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| liq_short_4h_oi | -0.146 | 35.357 |
| ret_24h | 0.291 | 28.462 |
| oi_z_7d | 1.887 | 14.260 |
| volatility_24h | -0.249 | 4.975 |
| ls_long_pct | -2.355 | 3.595 |
| volume_ratio_24h | 0.251 | 2.482 |
| volume_z_7d | 1.099 | 1.668 |
| funding | 0.477 | 1.592 |
| oi_chg_24h | 0.352 | 1.001 |
| oi_z_24h | -0.525 | 0.892 |
| oi_chg_4h | -0.446 | 0.820 |
| funding_z_7d | 0.012 | 0.813 |
| ls_chg_24h | 0.715 | 0.769 |
| buy_ratio_4h | 0.414 | 0.634 |
| liq_long_4h_oi | -0.305 | 0.522 |
| oi_chg_1h | -0.077 | 0.402 |
| oi_minus_ret_24h | 0.207 | 0.368 |
| range_1h | 0.135 | 0.368 |
| ret_1h | 0.254 | 0.205 |
| funding_chg_24h | -0.505 | 0.193 |
| liq_imbalance_4h | 0.293 | 0.185 |
| pred_funding_gap | -0.027 | 0.170 |
| ret_4h | 0.474 | 0.149 |
| buy_ratio_1h | -0.268 | 0.117 |
