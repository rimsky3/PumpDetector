# Training report

Generated 2026-09-29 12:53 UTC. Leave-ticker-out validation: every ticker is scored by a model that never saw it; the probability threshold is chosen by nested leave-ticker-out on the training tickers only.

Data: 12758 hourly bars, 10 tickers (AIN, AKE, G, LSK, MUBARAK, MYX, NMR, QNT, SOON, TRUMP), 461 pump hours (3.6%), 21 pump intervals.

## Summary

Hour level: precision / recall / F1 of the `pump in progress` class; PR-AUC is threshold-free. Event level: pumps with at least one signal inside, median hours from pump start to the first signal, false starts that triggered a signal, and signal episodes outside any labeled interval per 30 days of one ticker.

| model | precision | recall | f1 | pr_auc | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|
| rule | 0.465 | 0.642 | 0.539 | — | 20/21 | 0.000 | 6/8 | 3.834 |
| logreg | 0.621 | 0.703 | 0.659 | 0.663 | 21/21 | 0.000 | 5/8 | 2.707 |
| lightgbm | 0.593 | 0.766 | 0.669 | 0.702 | 19/21 | 0.000 | 5/8 | 1.748 |

## Training settings

- `--fresh-hours 48`: 402 unlabeled hours after 2026-09-27 12:53 UTC left out of training
- models: rule, logreg, lightgbm; 24 features; saved for the screener: yes

## rule by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.780 | 1/1 | 0.000 | 0/0 | 4.074 |
| AKE | 0.652 | 5/6 | 4.000 | 0/1 | 1.090 |
| G | 0.290 | 1/1 | 0.000 | 1/2 | 3.815 |
| LSK | 0.432 | 1/1 | 0.000 | 2/2 | 2.725 |
| MUBARAK | 0.769 | 5/5 | 0.000 | 1/1 | 4.660 |
| MYX | 0.062 | 1/1 | 0.000 | 2/2 | 3.815 |
| NMR | 0.562 | 2/2 | 0.000 | 0/0 | 6.236 |
| QNT | 0.312 | 1/1 | 0.000 | 0/0 | 4.038 |
| SOON | 0.519 | 1/1 | 0.000 | 0/0 | 2.832 |
| TRUMP | 0.414 | 2/2 | 0.500 | 0/0 | 5.298 |

## logreg by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.872 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.613 | 6/6 | 0.500 | 0/1 | 13.626 |
| G | 0.450 | 1/1 | 0.000 | 1/2 | 2.180 |
| LSK | 0.449 | 1/1 | 1.000 | 2/2 | 4.360 |
| MUBARAK | 0.839 | 5/5 | 0.000 | 1/1 | 0.583 |
| MYX | 0.444 | 1/1 | 0.000 | 1/2 | 0.545 |
| NMR | 0.919 | 2/2 | 0.500 | 0/0 | 0.567 |
| QNT | 0.559 | 1/1 | 0.000 | 0/0 | 4.615 |
| SOON | 0.600 | 1/1 | 0.000 | 0/0 | 0.000 |
| TRUMP | 0.700 | 2/2 | 2.000 | 0/0 | 0.000 |

## lightgbm by ticker

| ticker | f1 | pumps_caught | median_delay_h | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|
| AIN | 0.965 | 1/1 | 0.000 | 0/0 | 0.000 |
| AKE | 0.770 | 5/6 | 3.000 | 0/1 | 0.545 |
| G | 0.300 | 1/1 | 0.000 | 1/2 | 1.635 |
| LSK | 0.641 | 1/1 | 0.000 | 2/2 | 5.450 |
| MUBARAK | 0.643 | 5/5 | 0.000 | 1/1 | 8.155 |
| MYX | 0.267 | 1/1 | 0.000 | 1/2 | 1.090 |
| NMR | 0.643 | 2/2 | 2.000 | 0/0 | 0.000 |
| QNT | 0.639 | 1/1 | 0.000 | 0/0 | 0.000 |
| SOON | 0.698 | 1/1 | 1.000 | 0/0 | 0.000 |
| TRUMP | 0.235 | 1/2 | 2.000 | 0/0 | 0.589 |

## Feature importance

Models fitted on all data. Logistic regression coefficients are on standardised features (sign = direction of effect). LightGBM gain is the share of total split gain.

| feature | logreg_coef | lightgbm_gain_% |
|---|---|---|
| ret_24h | 0.312 | 44.590 |
| liq_short_4h_oi | -0.184 | 17.949 |
| oi_z_7d | 1.867 | 14.928 |
| volatility_24h | -0.236 | 4.776 |
| ls_long_pct | -2.177 | 4.355 |
| volume_ratio_24h | 0.255 | 2.805 |
| volume_z_7d | 1.032 | 1.860 |
| funding | 0.523 | 1.192 |
| oi_z_24h | -0.534 | 1.151 |
| oi_chg_4h | -0.389 | 1.002 |
| oi_chg_1h | -0.037 | 0.825 |
| ls_chg_24h | 0.739 | 0.777 |
| oi_chg_24h | 0.362 | 0.734 |
| buy_ratio_4h | 0.429 | 0.635 |
| funding_z_7d | 0.012 | 0.462 |
| liq_long_4h_oi | -0.116 | 0.400 |
| oi_minus_ret_24h | 0.199 | 0.365 |
| funding_chg_24h | -0.529 | 0.348 |
| liq_imbalance_4h | 0.315 | 0.199 |
| range_1h | 0.183 | 0.193 |
| ret_4h | 0.478 | 0.149 |
| pred_funding_gap | -0.021 | 0.144 |
| ret_1h | 0.236 | 0.084 |
| buy_ratio_1h | -0.273 | 0.077 |
