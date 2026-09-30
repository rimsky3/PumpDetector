# Experiment: 15-minute model with liquidations and short features

14 tickers from tickers.txt, 2026-08-07 – 2026-09-30, 49 labeled pumps (impulse labels on 5m/15m). 15m data from the database: Binance + Bybit candles, OI, funding, long/short, and Coinalyze 15m liquidations, known on 99% of bars (Coinalyze keeps a limited number of points; before a market's coverage starts, liquidations are missing, not zero). Leave-ticker-out with nested threshold.

- **1h (current)**: production features on hourly Coinalyze data, same window
- **base**: 20 features without liquidations, 15m bars, same wall-clock windows
- **+liq**: all production features, liquidations from 15m data
- **+short**: base + ret_bar, ret_2bar, oi_chg_bar, oi_chg_2bar, oi_accel_1h, ret_accel_1h, vol_burst, buy_ratio_bar, range_bar
- **+liq+short**: everything, plus liq_long_1h_oi, liq_short_1h_oi

Event metrics as in `experiments/timeframe.py`: delay to the close of the first signal bar, share of the move already gone, peak / signal price − 1, false episodes (bars ≤ 1 h apart) per ticker-month.

|  | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | median_delay_min | median_move_missed | median_upside_left | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|---|---|
| rule · 1h (current) | 0.358 | 0.556 | 0.436 | — | 36/49 | 5/19 | 105.000 | 0.411 | 0.196 | 3.524 |
| logreg · 1h (current) | 0.375 | 0.546 | 0.444 | 0.390 | 42/49 | 4/19 | 95.000 | 0.406 | 0.170 | 4.793 |
| lightgbm · 1h (current) | 0.437 | 0.612 | 0.510 | 0.453 | 45/49 | 4/19 | 95.000 | 0.423 | 0.160 | 4.135 |
| rule · base | 0.393 | 0.519 | 0.447 | — | 39/49 | 5/19 | 40.000 | 0.232 | 0.243 | 4.977 |
| logreg · base | 0.353 | 0.526 | 0.422 | 0.335 | 47/49 | 13/19 | 30.000 | 0.199 | 0.241 | 7.606 |
| lightgbm · base | 0.440 | 0.599 | 0.508 | 0.439 | 46/49 | 12/19 | 30.000 | 0.208 | 0.233 | 6.291 |
| logreg · +liq | 0.348 | 0.521 | 0.417 | 0.341 | 47/49 | 13/19 | 25.000 | 0.167 | 0.243 | 7.935 |
| lightgbm · +liq | 0.464 | 0.580 | 0.516 | 0.461 | 44/49 | 12/19 | 27.500 | 0.216 | 0.258 | 5.775 |
| logreg · +short | 0.348 | 0.542 | 0.424 | 0.336 | 47/49 | 13/19 | 25.000 | 0.199 | 0.241 | 7.747 |
| lightgbm · +short | 0.449 | 0.610 | 0.517 | 0.455 | 48/49 | 12/19 | 27.500 | 0.208 | 0.211 | 6.291 |
| logreg · +liq+short | 0.343 | 0.544 | 0.421 | 0.337 | 47/49 | 14/19 | 25.000 | 0.167 | 0.243 | 7.888 |
| lightgbm · +liq+short | 0.466 | 0.580 | 0.517 | 0.466 | 47/49 | 11/19 | 25.000 | 0.221 | 0.243 | 5.258 |

## LightGBM gain, all 15m features (fitted on all data)

|  | lightgbm_gain_% |
|---|---|
| oi_chg_24h | 48.921 |
| ret_24h | 11.912 |
| volume_z_7d | 7.058 |
| oi_z_24h | 6.664 |
| funding_z_7d | 2.953 |
| volume_ratio_24h | 2.685 |
| ls_long_pct | 2.590 |
| ret_4h | 2.568 |
| oi_minus_ret_24h | 1.972 |
| oi_z_7d | 1.950 |
| oi_chg_4h | 1.387 |
| oi_chg_1h | 1.238 |
| liq_long_4h_oi | 1.231 |
| volatility_24h | 1.212 |
| ls_chg_24h | 0.985 |
