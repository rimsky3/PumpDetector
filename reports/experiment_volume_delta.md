# Experiment: aggregated volume delta (perpetuals and spot)

12806 hourly bars, 10 tickers, 461 pump hours. Leave-ticker-out with nested threshold, as in train.py. Perp delta from Coinalyze buy volume across exchanges; spot delta from Binance spot (available for G, LSK, MUBARAK, NMR, QNT, TRUMP; other tickers have these features empty).

- **base**: current features (23)
- **+perp**: cvd_24h, vd_1h_oi, cvd_24h_oi, vd_z_7d
- **+spot**: spot_buy_ratio_4h, spot_cvd_24h, spot_vd_z_7d, spot_perp_vol_24h, spot_share_z_7d
- **+both**: all of them

## Summary

|  | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|
| logreg · base | 0.622 | 0.722 | 0.669 | 0.675 | 21/21 | 5/8 | 2.810 |
| lightgbm · base | 0.597 | 0.753 | 0.666 | 0.701 | 18/21 | 5/8 | 1.686 |
| logreg · +perp | 0.602 | 0.731 | 0.660 | 0.650 | 21/21 | 5/8 | 2.473 |
| lightgbm · +perp | 0.589 | 0.759 | 0.664 | 0.720 | 20/21 | 5/8 | 1.686 |
| logreg · +spot | 0.579 | 0.690 | 0.630 | 0.666 | 21/21 | 5/8 | 2.641 |
| lightgbm · +spot | 0.644 | 0.735 | 0.687 | 0.708 | 19/21 | 5/8 | 1.293 |
| logreg · +both | 0.556 | 0.690 | 0.616 | 0.646 | 21/21 | 5/8 | 2.810 |
| lightgbm · +both | 0.635 | 0.731 | 0.679 | 0.714 | 20/21 | 5/8 | 1.742 |

## LightGBM gain of the new features (+both, fitted on all data)

|  | lightgbm_gain_% |
|---|---|
| cvd_24h | 0.675 |
| vd_1h_oi | 0.000 |
| cvd_24h_oi | 2.341 |
| vd_z_7d | 0.212 |
| spot_buy_ratio_4h | 0.423 |
| spot_cvd_24h | 0.142 |
| spot_vd_z_7d | 0.814 |
| spot_perp_vol_24h | 0.584 |
| spot_share_z_7d | 0.828 |
