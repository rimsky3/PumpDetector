# Screener and dashboard

A working skeleton of the live product: an hourly screener that scores a watchlist with the trained models,
and a dashboard with two screens, Forecast and Screener, built from the "Pump Detector App" design mockup.

```
train.py ──saves──▶ models/logreg.joblib, models/lightgbm.joblib   (model + threshold + feature list)

screener.py, every hour at HH:05 UTC:
    fetch the last 10 days for watchlist.txt  →  features.build()  →  both models  →  signals table

dashboard.py + app/dashboard.html:  signals + raw series + risk  →  dashboard
```

## Running it

```bash
python train.py                        # trains, validates and saves models/ (needed once, then after changes)
python screener.py                     # one cycle
python screener.py --loop              # keep running, one cycle per hour
python screener.py --no-fetch          # score data already in the database, no API calls
python dashboard.py                    # dashboard at http://127.0.0.1:8766
```

## Watchlist

`watchlist.txt` lists the coins to scan, in the same format as `tickers.txt`. The two files are separate on
purpose:

| File | Role |
|---|---|
| `tickers.txt` | training set: labeled tickers only; `features.py` and `train.py` use these |
| `watchlist.txt` | anything to scan live; never used for training |

The screener writes raw data for watchlist coins into the same tables. If unlabeled coins reached the
training set, every hour on them would count as "no pump" and silently bias the model. That is why
`features.py` builds the training table from `tickers.txt` only.

**Capacity.** Coinalyze allows 40 symbol-requests per minute. A ticker has about 3 exchange contracts and
6 series, so roughly 18 calls. Around 100 tickers fit into an hourly cycle. A coin needs 7 days of
history before it can be scored, because the baseline windows are 7 days.

## Models

`train.py` saves, for each probability model, a model fitted on all labeled data, together with:

- the feature list (the screener refuses a model trained on a different feature set);
- the signal threshold, chosen as the best F1 on out-of-fold predictions, so it was never tuned on data the
  scored model had seen;
- training time, tickers, last training hour and the leave-ticker-out validation metrics.

Both models run side by side. This is a forward test: over the coming weeks, live data shows which one
actually works better.

## `signals` table

One row per ticker × closed hour × model version.

| Column | Meaning |
|---|---|
| `probability`, `threshold`, `signal` | model output, its threshold, and whether it fired |
| `in_sample` | this hour of this ticker was in the model's training data, so the score is recall, not prediction |
| `live` | the hour was the latest closed bar when scored and was not in-sample: an honest real-time prediction |
| `model_version` | when the model was trained; retraining starts a new version |

The first score of an hour is never overwritten (`ON CONFLICT DO NOTHING`), so the record of what the model
said in real time cannot be rewritten after the fact. On first run the screener also backfills the last
7 days (`live = false`), so the dashboard has history to show.

**Only `live` rows count as a forward test.** For tickers in the training set, every hour up to the training
cut-off is in-sample: a high probability there means the model remembers the pump, not that it predicted it.

## Dashboard

`python dashboard.py` serves `app/dashboard.html` on 127.0.0.1 (this computer only). Plain Python server,
no extra dependencies. `#forecast/NMR` in the address opens a ticker directly.

**Forecast** (one ticker):

- current pump probability with the model's threshold and status, OI change over 4 hours, volume z-score;
- risk cards: stop, exit signal and trailing stop (see Risk below) for the current or the last trade;
- chart: price with hours above the threshold shaded, entry point, stop and trailing lines, and the
  probability per hour underneath;
- **Прогнозувати** fetches the latest data for this ticker from Coinalyze and scores it again with both models
  (about 18 API calls, a few seconds).

**Screener** (all scored tickers): price sparkline for 48 hours, probability, status and the signal log.

| Status | Rule |
|---|---|
| Alert | probability ≥ the model's threshold |
| Кандидат | probability ≥ half the threshold |
| Спокійно | below that |

The model selector in the sidebar switches every screen between LightGBM and logistic regression.

## Risk

`risk.py` implements the exit plan. `train.py` computes the stop multiplier and saves it with each model.

| Part | Rule |
|---|---|
| NATR | ATR(14), Wilder smoothing, on hourly bars |
| Stop | entry − k·ATR at entry; k = 75th percentile of MAE of labeled pumps, in ATR units |
| Trailing stop | highest high since entry − 1.5·ATR now |
| Exit | probability below the threshold for 2 bars in a row, or the stop is hit |

**MAE** (maximum adverse excursion) of a pump is how far the price fell below the entry before reaching the
peak. Entry is the close of the first labeled bar. The fall after the peak is not included: that is the
exit, not a pullback the stop had to survive. On 18 pumps: k = 2.9, median MAE 1.1 ATR, median move to the
peak 18.8 ATR.

A "trade" on the dashboard starts at the first hour with probability ≥ threshold and ends by the exit rule.
It is a simulation to judge the model, not trading advice.

## Next steps

- Coins outside the training set in `watchlist.txt`, so live scores are genuine out-of-sample.
- Forward-test report from `live` rows once a few weeks accumulate: signals vs what happened.
- Telegram notifications for new signals (the mockup's signal log already has room for them).
- Deployment to the VPS with Docker Compose: database, screener loop and dashboard as services.
