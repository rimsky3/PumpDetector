# Experiment: 15m bars vs 1h

Binance + Bybit public APIs, 10 tickers, 2026-08-07 – 2026-09-29, 21 labeled pumps. Both models use the same 20 features with the same wall-clock windows (no liquidations or predicted funding: exchanges keep no history of them), leave-ticker-out with nested threshold as in train.py. The only difference is how often the model re-scores: every hour or every 15m.

- **precision / recall / f1 / pr_auc** are per bar, so they are not directly comparable across timeframes
- **median_delay_min**: from the labeled pump start to the first signal, counted at bar close (when it can be acted on)
- **median_move_missed**: share of the move from start to peak already gone at the signal price
- **median_upside_left**: peak / signal price − 1
- **false_alarms_per_30d**: signal episodes (bars ≤ 1 h apart) that overlap no label, per ticker-month

|  | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | median_delay_min | median_move_missed | median_upside_left | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|---|---|---|
| rule · 1h | 0.495 | 0.657 | 0.565 | — | 20/21 | 6/8 | 60.000 | 0.224 | 0.513 | 3.651 |
| logreg · 1h | 0.656 | 0.694 | 0.674 | 0.682 | 20/21 | 5/8 | 60.000 | 0.218 | 0.497 | 3.120 |
| lightgbm · 1h | 0.594 | 0.714 | 0.648 | 0.696 | 20/21 | 5/8 | 120.000 | 0.299 | 0.419 | 2.058 |
| rule · 15m | 0.531 | 0.701 | 0.604 | — | 20/21 | 7/8 | 37.500 | 0.116 | 0.537 | 5.639 |
| logreg · 15m | 0.630 | 0.721 | 0.672 | 0.678 | 20/21 | 5/8 | 45.000 | 0.132 | 0.571 | 5.042 |
| lightgbm · 15m | 0.584 | 0.666 | 0.622 | 0.612 | 19/21 | 6/8 | 45.000 | 0.116 | 0.686 | 2.255 |
