# Experiment: open interest in coins vs USD

11871 hourly bars, 9 tickers, 448 pump hours. Leave-ticker-out with nested threshold selection, as in train.py.

- **usd** — current features (OI in USD as provided by Coinalyze)
- **coins** — OI features (changes, z-scores) computed on OI in coins; `oi_minus_ret_24h` dropped
- **both** — current features plus the coin-based ones

## Summary

| model · variant | precision | recall | f1 | pr_auc | pumps_caught | false_starts_triggered | false_alarms_per_30d |
|---|---|---|---|---|---|---|---|
| rule · usd | 0.473 | 0.705 | 0.566 | — | 18/19 | 6/8 | 3.824 |
| logreg · usd | 0.581 | 0.692 | 0.631 | 0.643 | 18/19 | 5/8 | 2.792 |
| lightgbm · usd | 0.619 | 0.761 | 0.683 | 0.709 | 18/19 | 4/8 | 1.882 |
| rule · coins | 0.427 | 0.571 | 0.489 | — | 16/19 | 6/8 | 4.552 |
| logreg · coins | 0.601 | 0.656 | 0.628 | 0.573 | 18/19 | 4/8 | 1.700 |
| lightgbm · coins | 0.576 | 0.661 | 0.615 | 0.660 | 17/19 | 4/8 | 2.185 |
| logreg · both | 0.626 | 0.679 | 0.651 | 0.642 | 18/19 | 5/8 | 2.914 |
| lightgbm · both | 0.581 | 0.730 | 0.647 | 0.650 | 18/19 | 3/8 | 1.821 |

## F1 by ticker

| model · variant | AIN | AKE | G | LSK | MUBARAK | MYX | NMR | QNT | SOON |
|---|---|---|---|---|---|---|---|---|---|
| rule · usd | 0.780 | 0.697 | 0.290 | 0.500 | 0.769 | 0.062 | 0.419 | 0.468 | 0.519 |
| logreg · usd | 0.872 | 0.576 | 0.462 | 0.421 | 0.806 | 0.500 | 0.694 | 0.523 | 0.600 |
| lightgbm · usd | 0.952 | 0.761 | 0.300 | 0.752 | 0.664 | 0.500 | 0.400 | 0.602 | 0.690 |
| rule · coins | 0.777 | 0.167 | 0.321 | 0.438 | 0.762 | 0.062 | 0.424 | 0.444 | 0.571 |
| logreg · coins | 0.989 | 0.536 | 0.340 | 0.479 | 0.818 | 0.267 | 0.652 | 0.556 | 0.610 |
| lightgbm · coins | 0.927 | 0.320 | 0.346 | 0.754 | 0.655 | 0.571 | 0.462 | 0.645 | 0.787 |
| logreg · both | 0.940 | 0.621 | 0.419 | 0.422 | 0.802 | 0.571 | 0.694 | 0.534 | 0.600 |
| lightgbm · both | 1.000 | 0.592 | 0.305 | 0.724 | 0.643 | 0.364 | 0.378 | 0.656 | 0.690 |
