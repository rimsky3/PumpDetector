# Training report

Generated 2026-09-29 19:09 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 12814 hourly bars, 10 tickers (AIN, AKE, G, LSK, MUBARAK, MYX, NMR, QNT, SOON, TRUMP), 461 pump hours (3.6%), 21 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.464 | 0.642 | 0.539 | — | 20/21 | 0.000 | 6/8 | 3.876 |
| logreg | 0.622 | 0.722 | 0.669 | 0.675 | 21/21 | 0.000 | 5/8 | 2.809 |
| lightgbm | 0.601 | 0.777 | 0.677 | 0.704 | 19/21 | 0.000 | 4/8 | 2.022 |

## Training settings

- `--fresh-hours 48`: 396 unlabeled hours after 2026-09-27 19:08 UTC left out of training
- models: rule, logreg, lightgbm; 23 features; saved for the screener: no

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.780 | 1/1 | 0.000 | 0/0 | 4.051 |
| AKE | 0.652 | 5/6 | 4.000 | 0/1 | 1.084 |
| G | 0.290 | 1/1 | 0.000 | 1/2 | 3.795 |
| LSK | 0.432 | 1/1 | 0.000 | 2/2 | 2.711 |
| MUBARAK | 0.769 | 5/5 | 0.000 | 1/1 | 4.634 |
| MYX | 0.062 | 1/1 | 0.000 | 2/2 | 3.795 |
| NMR | 0.554 | 2/2 | 0.000 | 0/0 | 6.803 |
| QNT | 0.312 | 1/1 | 0.000 | 0/0 | 4.038 |
| SOON | 0.519 | 1/1 | 0.000 | 0/0 | 2.832 |
| TRUMP | 0.414 | 2/2 | 0.500 | 0/0 | 5.268 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.886 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.624 | 6/6 | 0.500 | 0/1 | 14.639 |
| G | 0.450 | 1/1 | 0.000 | 1/2 | 2.169 |
| LSK | 0.484 | 1/1 | 1.000 | 2/2 | 4.337 |
| MUBARAK | 0.839 | 5/5 | 0.000 | 1/1 | 0.579 |
| MYX | 0.444 | 1/1 | 0.000 | 1/2 | 0.542 |
| NMR | 0.895 | 2/2 | 0.500 | 0/0 | 0.567 |
| QNT | 0.585 | 1/1 | 0.000 | 0/0 | 4.615 |
| SOON | 0.600 | 1/1 | 0.000 | 0/0 | 0.000 |
| TRUMP | 0.700 | 2/2 | 2.000 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.977 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.777 | 5/6 | 3.000 | 0/1 | 0.542 |
| G | 0.327 | 1/1 | 0.000 | 0/2 | 3.253 |
| LSK | 0.661 | 1/1 | 0.000 | 2/2 | 4.337 |
| MUBARAK | 0.634 | 5/5 | 0.000 | 1/1 | 9.847 |
| MYX | 0.286 | 1/1 | 0.000 | 1/2 | 1.084 |
| NMR | 0.643 | 2/2 | 2.000 | 0/0 | 0.000 |
| QNT | 0.656 | 1/1 | 0.000 | 0/0 | 0.577 |
| SOON | 0.698 | 1/1 | 1.000 | 0/0 | 0.000 |
| TRUMP | 0.235 | 1/2 | 2.000 | 0/0 | 0.585 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.313 | 39.757 |
| liq_short_4h_oi | -0.188 | 17.763 |
| oi_z_7d | 1.880 | 16.635 |
| volatility_24h | -0.240 | 4.926 |
| ls_long_pct | -2.190 | 4.402 |
| oi_chg_24h | 0.359 | 3.796 |
| volume_ratio_24h | 0.260 | 3.447 |
| volume_z_7d | 1.030 | 1.654 |
| funding | 0.534 | 1.287 |
| oi_z_24h | -0.542 | 1.275 |
| oi_chg_4h | -0.384 | 0.836 |
| ls_chg_24h | 0.743 | 0.656 |
| buy_ratio_4h | 0.431 | 0.630 |
| oi_chg_1h | -0.036 | 0.626 |
| funding_z_7d | 0.012 | 0.544 |
| oi_minus_ret_24h | 0.193 | 0.409 |
| liq_long_4h_oi | -0.110 | 0.391 |
| funding_chg_24h | -0.538 | 0.290 |
| liq_imbalance_4h | 0.320 | 0.209 |
| range_1h | 0.185 | 0.166 |
| ret_4h | 0.476 | 0.154 |
| ret_1h | 0.238 | 0.075 |
| buy_ratio_1h | -0.274 | 0.074 |
