# Experiment: exchange APIs directly vs Coinalyze

Tickers: AIN, AKE, G, LSK, MUBARAK, MYX, NMR, QNT, SOON, TRUMP. Direct data: Binance and Bybit USDT perpetuals (18 markets), public endpoints, no API key.
Comparison window: from 2026-09-08 04:00 UTC (4774 hourly bars, 10 labeled pumps). Binance serves open interest and long/short history for the last 30 days only; features need 7 days of warm-up.

- **coinalyze**: production features, all exchanges
- **coinalyze_bb**: Coinalyze restricted to Binance and Bybit, separating the source effect from exchange coverage
- **direct_hybrid**: Binance and Bybit APIs; liquidations still from Coinalyze (exchanges keep no history of them)
- **direct**: exchange APIs only; no liquidations (their features are 0)

## 1. Raw series, contract by contract

Correlation of hourly changes with the exchange series shifted by −1/0/+1 h: the peak at 0 confirms the time alignment. Exchange OI snapshots at T+1h are the close of Coinalyze's bar T; long/short snapshots at T and funding paid at T match bar T.

|  | hours | median_rel_diff | corr_chg_shift_-1h | corr_chg_shift_+0h | corr_chg_shift_+1h |
|---|---|---|---|---|---|
| ('Binance', 'close') | 14400.000 | 0.000 | -0.067 | 1.000 | -0.067 |
| ('Binance', 'volume') | 14400.000 | 0.000 | -0.171 | 1.000 | -0.171 |
| ('Binance', 'buy volume') | 14400.000 | 0.000 | -0.168 | 1.000 | -0.168 |
| ('Binance', 'OI (USD)') | 7080.000 | 0.000 | 0.020 | 1.000 | 0.020 |
| ('Binance', 'funding') | 14400.000 | 0.000 | -0.078 | 1.000 | -0.078 |
| ('Binance', 'long %') | 7080.000 | 0.000 | 0.060 | 1.000 | 0.060 |
| ('Bybit', 'close') | 11520.000 | 0.000 | -0.068 | 1.000 | -0.068 |
| ('Bybit', 'volume') | 11520.000 | 0.000 | -0.247 | 1.000 | -0.247 |
| ('Bybit', 'OI (USD)') | 11512.000 | 0.000 | 0.062 | 1.000 | 0.062 |
| ('Bybit', 'funding') | 11520.000 | 0.000 | 0.082 | 1.000 | 0.082 |
| ('Bybit', 'long %') | 11520.000 | 0.000 | -0.070 | 1.000 | -0.070 |

## 2. Features

Spearman correlation of each feature with the production (coinalyze) value on the same hours.

|  | spearman vs coinalyze · coinalyze_bb | spearman vs coinalyze · direct_hybrid | spearman vs coinalyze · direct | NaN share · direct |
|---|---|---|---|---|
| oi_chg_1h | 0.993 | 0.993 | 0.993 | 0.000 |
| oi_chg_4h | 0.994 | 0.994 | 0.994 | 0.000 |
| oi_chg_24h | 0.996 | 0.996 | 0.996 | 0.000 |
| oi_z_24h | 0.994 | 0.994 | 0.994 | 0.000 |
| oi_z_7d | 0.996 | 0.996 | 0.996 | 0.000 |
| ret_1h | 1.000 | 1.000 | 1.000 | 0.000 |
| ret_4h | 1.000 | 1.000 | 1.000 | 0.000 |
| ret_24h | 1.000 | 1.000 | 1.000 | 0.000 |
| volatility_24h | 1.000 | 1.000 | 1.000 | 0.000 |
| range_1h | 1.000 | 1.000 | 1.000 | 0.000 |
| oi_minus_ret_24h | 0.982 | 0.982 | 0.982 | 0.000 |
| volume_z_7d | 0.996 | 0.996 | 0.996 | 0.000 |
| volume_ratio_24h | 0.998 | 0.998 | 0.998 | 0.000 |
| buy_ratio_1h | 0.962 | 0.888 | 0.888 | 0.000 |
| buy_ratio_4h | 0.962 | 0.898 | 0.898 | 0.000 |
| funding | 0.949 | 0.943 | 0.943 | 0.000 |
| funding_z_7d | 0.914 | 0.920 | 0.920 | 0.000 |
| funding_chg_24h | 0.957 | 0.955 | 0.955 | 0.000 |
| liq_long_4h_oi | 0.994 | 0.994 | — | 0.000 |
| liq_short_4h_oi | 0.988 | 0.988 | — | 0.000 |
| liq_imbalance_4h | 0.971 | 0.971 | — | 0.000 |
| ls_long_pct | 1.000 | 1.000 | 1.000 | 0.000 |
| ls_chg_24h | 1.000 | 1.000 | 1.000 | 0.000 |

## 3. Can the current model take exchange data?

Models trained on Coinalyze features (60 days, leave-ticker-out, nested threshold as in train.py) and scored on the held-out ticker's features from each source.

|  | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|
| rule · coinalyze | 0.559 | 0.577 | 0.568 | — | 10/10 | 1/3 | 2.863 |
| rule · coinalyze_bb | 0.558 | 0.571 | 0.564 | — | 10/10 | 1/3 | 2.863 |
| rule · direct_hybrid | 0.558 | 0.571 | 0.564 | — | 10/10 | 1/3 | 2.863 |
| rule · direct | 0.558 | 0.571 | 0.564 | — | 10/10 | 1/3 | 2.863 |
| logreg · coinalyze | 0.747 | 0.680 | 0.712 | 0.740 | 10/10 | 1/3 | 2.863 |
| logreg · coinalyze_bb | 0.740 | 0.675 | 0.706 | 0.733 | 10/10 | 1/3 | 2.712 |
| logreg · direct_hybrid | 0.740 | 0.675 | 0.706 | 0.733 | 10/10 | 1/3 | 2.712 |
| logreg · direct | 0.733 | 0.689 | 0.710 | 0.742 | 10/10 | 1/3 | 2.712 |
| lightgbm · coinalyze | 0.669 | 0.766 | 0.714 | 0.750 | 10/10 | 0/3 | 3.014 |
| lightgbm · coinalyze_bb | 0.673 | 0.754 | 0.711 | 0.756 | 10/10 | 0/3 | 2.712 |
| lightgbm · direct_hybrid | 0.671 | 0.754 | 0.710 | 0.755 | 10/10 | 0/3 | 2.712 |
| lightgbm · direct | 0.892 | 0.438 | 0.587 | 0.783 | 9/10 | 0/3 | 0.904 |

## 4. Is exchange data as informative?

Leave-ticker-out trained and tested inside the comparison window only, separately for each source. Few pumps: differences of a few points are noise.

|  | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|
| logreg · coinalyze | 0.702 | 0.843 | 0.766 | 0.657 | 10/10 | 1/3 | 3.014 |
| lightgbm · coinalyze | 0.571 | 0.784 | 0.661 | 0.552 | 10/10 | 2/3 | 2.110 |
| logreg · coinalyze_bb | 0.666 | 0.843 | 0.744 | 0.658 | 10/10 | 1/3 | 3.617 |
| lightgbm · coinalyze_bb | 0.610 | 0.820 | 0.699 | 0.697 | 10/10 | 0/3 | 1.959 |
| logreg · direct_hybrid | 0.660 | 0.834 | 0.737 | 0.708 | 10/10 | 1/3 | 3.466 |
| lightgbm · direct_hybrid | 0.578 | 0.831 | 0.682 | 0.595 | 10/10 | 2/3 | 1.808 |
| logreg · direct | 0.684 | 0.802 | 0.738 | 0.817 | 10/10 | 1/3 | 3.466 |
| lightgbm · direct | 0.555 | 0.802 | 0.656 | 0.659 | 10/10 | 0/3 | 3.165 |
