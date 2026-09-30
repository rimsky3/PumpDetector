# Experiment: 15m bars vs 1h

Binance + Bybit public APIs, 14 tickers, 2026-08-08 – 2026-09-30, 49 labeled pumps. Both models use the same 20 features with the same wall-clock windows (no liquidations: exchanges keep no history of them), leave-ticker-out with nested threshold as in train.py. The only difference is how often the model re-scores: every hour or every 15m.

- **precision / recall / f1 / pr_auc** are per bar, so they are not directly comparable across timeframes
- **median_delay_min**: from the labeled pump start to the first signal, counted at bar close (when it can be acted on)
- **median_move_missed**: share of the move from start to peak already gone at the signal price
- **median_upside_left**: peak / signal price − 1
- **false_alarms_per_30d**: signal episodes (bars ≤ 1 h apart) that overlap no label, per ticker-month

|  | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | median_delay_min | median_move_missed | median_upside_left | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|---|---|
| rule · 1h | 0.340 | 0.566 | 0.425 | — | 38/49 | 5/18 | 105.000 | 0.411 | 0.196 | 3.996 |
| logreg · 1h | 0.343 | 0.515 | 0.412 | 0.351 | 44/49 | 6/18 | 97.500 | 0.452 | 0.130 | 5.233 |
| lightgbm · 1h | 0.429 | 0.591 | 0.497 | 0.444 | 46/49 | 6/18 | 97.500 | 0.425 | 0.160 | 4.853 |
| rule · 15m | 0.388 | 0.522 | 0.445 | — | 39/49 | 5/18 | 30.000 | 0.232 | 0.243 | 5.230 |
| logreg · 15m | 0.354 | 0.525 | 0.423 | 0.338 | 47/49 | 12/18 | 30.000 | 0.199 | 0.241 | 8.178 |
| lightgbm · 15m | 0.433 | 0.603 | 0.504 | 0.464 | 47/49 | 12/18 | 25.000 | 0.222 | 0.210 | 6.371 |
