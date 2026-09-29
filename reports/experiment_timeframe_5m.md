# Experiment: 5m bars vs 1h

Binance + Bybit public APIs, 10 tickers, 2026-08-07 – 2026-09-29, 21 labeled pumps. Both models use the same 20 features with the same wall-clock windows (no liquidations or predicted funding: exchanges keep no history of them), leave-ticker-out with nested threshold as in train.py. The only difference is how often the model re-scores: every hour or every 5m.

- **precision / recall / f1 / pr_auc** are per bar, so they are not directly comparable across timeframes
- **median_delay_min**: from the labeled pump start to the first signal, counted at bar close (when it can be acted on)
- **median_move_missed**: share of the move from start to peak already gone at the signal price
- **median_upside_left**: peak / signal price − 1
- **false_alarms_per_30d**: signal episodes (bars ≤ 1 h apart) that overlap no label, per ticker-month

|  | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | median_delay_min | median_move_missed | median_upside_left | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|---|---|
| rule · 1h | 0.495 | 0.657 | 0.565 | — | 20/21 | 6/8 | 60.000 | 0.210 | 0.544 | 3.649 |
| logreg · 1h | 0.656 | 0.694 | 0.674 | 0.682 | 20/21 | 5/8 | 60.000 | 0.216 | 0.504 | 3.118 |
| lightgbm · 1h | 0.575 | 0.707 | 0.634 | 0.695 | 20/21 | 5/8 | 120.000 | 0.267 | 0.427 | 2.853 |
| rule · 5m | 0.553 | 0.690 | 0.614 | — | 20/21 | 7/8 | 25.000 | 0.073 | 0.631 | 5.836 |
| logreg · 5m | 0.643 | 0.700 | 0.670 | 0.668 | 21/21 | 5/8 | 35.000 | 0.090 | 0.565 | 6.234 |
| lightgbm · 5m | 0.565 | 0.634 | 0.597 | 0.633 | 19/21 | 6/8 | 45.000 | 0.147 | 0.703 | 2.122 |
