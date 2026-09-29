# Experiment: open interest in coins vs USD

13160 hourly bars, 10 tickers, 461 pump hours. Leave-ticker-out with nested threshold selection, as in train.py.

- **usd** — current features (OI in USD as provided by Coinalyze)
- **coins** — OI features (changes, z-scores) computed on OI in coins; `oi_minus_ret_24h` dropped
- **both** — current features plus the coin-based ones

## Summary

| model · variant | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|
| rule · usd | 0.448 | 0.642 | 0.528 | — | 20/21 | 6/8 | 3.833 |
| logreg · usd | 0.616 | 0.685 | 0.649 | 0.664 | 21/21 | 5/8 | 2.409 |
| lightgbm · usd | 0.597 | 0.768 | 0.672 | 0.699 | 20/21 | 4/8 | 1.643 |
| rule · coins | 0.403 | 0.547 | 0.464 | — | 16/21 | 6/8 | 4.380 |
| logreg · coins | 0.639 | 0.668 | 0.653 | 0.594 | 21/21 | 5/8 | 2.190 |
| lightgbm · coins | 0.527 | 0.672 | 0.591 | 0.637 | 19/21 | 6/8 | 2.354 |
| logreg · both | 0.640 | 0.675 | 0.657 | 0.660 | 21/21 | 5/8 | 2.628 |
| lightgbm · both | 0.505 | 0.722 | 0.594 | 0.641 | 20/21 | 5/8 | 2.573 |

## F1 by ticker

| model · variant | AIN | AKE | G | LSK | MUBARAK | MYX | NMR | QNT | SOON | TRUMP |
|---|---|---|---|---|---|---|---|---|---|---|
| rule · usd | 0.780 | 0.652 | 0.290 | 0.432 | 0.769 | 0.062 | 0.409 | 0.312 | 0.519 | 0.414 |
| logreg · usd | 0.857 | 0.618 | 0.450 | 0.366 | 0.847 | 0.444 | 0.694 | 0.523 | 0.600 | 0.762 |
| lightgbm · usd | 0.977 | 0.761 | 0.295 | 0.724 | 0.624 | 0.571 | 0.412 | 0.677 | 0.667 | 0.333 |
| rule · coins | 0.777 | 0.031 | 0.321 | 0.438 | 0.762 | 0.062 | 0.414 | 0.444 | 0.554 | 0.353 |
| logreg · coins | 0.977 | 0.624 | 0.348 | 0.491 | 0.837 | 0.250 | 0.578 | 0.561 | 0.593 | 0.632 |
| lightgbm · coins | 0.965 | 0.298 | 0.277 | 0.662 | 0.605 | 0.400 | 0.585 | 0.716 | 0.745 | 0.333 |
| logreg · both | 0.900 | 0.682 | 0.419 | 0.313 | 0.848 | 0.400 | 0.667 | 0.533 | 0.565 | 0.700 |
| lightgbm · both | 0.977 | 0.585 | 0.281 | 0.627 | 0.553 | 0.118 | 0.564 | 0.692 | 0.674 | 0.400 |
