# Experiment: stop measured from the model's entry

Out-of-fold 15m LightGBM (leave-ticker-out, 14 tickers) with hourly confirmation (out-of-fold, ≥ 0.75 of threshold). Entry = first Alert bar inside each labeled pump.

Current k = 1.16 ATR: 75th percentile of drawdown from the **labeled start**, in ATR at the start.

## Drawdown from the model's entry (46 pumps with an Alert inside)

| | median | 75th | 90th |
|---|---|---|---|
| delay after start, min | 45 | 102 | 190 |
| move already gone at entry | 9.4% | 19.7% | 24.8% |
| ATR(15m) at entry, % of price | 2.54% | 3.78% | 4.90% |
| drawdown after entry, ATR | 0.55 | 1.44 | 2.51 |
| move from entry to peak, ATR | 10.70 | 21.88 | 39.28 |

## Trades on all out-of-fold Alerts by stop multiplier

Entry on the first Alert bar, exit when the Alert is gone for 4 bars, trailing 1.5 ATR; no fees or slippage. `avg_in_pump` / `avg_outside` split trades by whether they overlap a labeled pump.

| k | trades | win_% | stops_% | avg_% | median_% | avg_in_pump_% | avg_outside_% |
|---|---|---|---|---|---|---|---|
| 1.16 | 207.000 | 32.400 | 62.300 | 3.660 | -1.410 | 12.000 | -0.690 |
| 1.44 | 182.000 | 40.100 | 52.200 | 4.690 | -1.470 | 14.310 | -0.520 |
| 1.5 | 178.000 | 40.400 | 49.400 | 4.800 | -1.450 | 14.530 | -0.530 |
| 2.0 | 166.000 | 44.000 | 38.000 | 5.180 | -1.340 | 14.810 | -0.420 |
| 2.5 | 160.000 | 45.000 | 30.000 | 5.340 | -1.000 | 15.320 | -0.490 |
| 2.51 | 160.000 | 45.000 | 30.000 | 5.320 | -1.000 | 15.310 | -0.520 |
| 3.0 | 158.000 | 45.600 | 24.100 | 5.350 | -0.790 | 15.900 | -0.610 |
| 4.0 | 153.000 | 47.700 | 9.800 | 5.570 | -0.260 | 15.730 | -0.460 |
