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
5 series, so roughly 15 calls. Around 100 tickers fit into an hourly cycle. A coin needs 7 days of
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
no extra dependencies. `#forecast/NMR` opens a ticker, `#journal` the journal. Every block has a **?** button
with a short explanation.

**Forecast** — three columns:

- *left*: model switch; pump probability with a threshold bar and the signal's age in hours; OI change over
  4 h, volume z-score, share of long accounts; the model's simulated position with stop, trailing stop and exit
  state;
- *centre*: chart ([TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts),
  Apache 2.0, local copy in `app/vendor/`): candles with signal zones, entry and exit markers of every
  simulated trade, stop and trailing lines, OI candles (in coins by default, USD on a toggle), volume and the
  hourly probability with the threshold;
- *right*: the watchlist like a trading terminal — search, **Прогнозувати**, filters (all / Alert /
  candidates / ★ favourites), sorting by probability, OI change over 24 h or price change over 24 h, a 24-hour
  probability sparkline per ticker, and the latest signal episodes.

**Ruler** (like TradingView's measure tool; shared `app/js/ruler.js`, also in the labeling tool): Shift + click on the
chart or the ruler button, move, click to fix, click again or Esc to remove. Shows the price change, bars and
time, summed volume, OI change in USD and in coins, and the model's max probability with hours above threshold.

Keyboard: **↑/↓** next ticker, **Enter** refresh the ticker, **F** favourite. **Alert notifications**: when a new
live signal episode starts while the page is open, a sound, a browser notification and an in-page toast
(opt-in; favourites, sort order and settings are stored in the browser).

| Status | Rule |
|---|---|
| Alert | probability ≥ the model's threshold |
| Кандидат | probability ≥ half the threshold |
| Спокійно | below that |

**Journal** — forward-test of what the models said in real time (`journal.py`):

- every hour's **first** score is used (across model versions), so retraining never rewrites history;
- trades are simulated with the rules below; each trade's outcome comes from the labels when they exist
  (`pumps.csv`: overlaps a pump → pump, a false start → false; for training tickers, hours up to the end of the
  training data are fully labeled), otherwise from an **auto rule**: within the horizon after entry, price
  and OI rose at least by the thresholds (default 24 h, +15 %, +20 %; editable on the page). Until the horizon
  passes the outcome is *waiting*;
- scope: live only (honest), + history backfill, + in-sample; period 7 / 30 days / all;
- per model: signals, share that turned out pumps (precision), labeled pumps caught, median delay, average
  trade, false signals per 30 ticker-days — next to the leave-ticker-out validation for comparison;
- trade results chart, by-ticker hit rate, how many trades are confirmed by labels, and system status
  (last scan, Coinalyze calls and 429s from `reports/screener_status.json`, tickers short of history).

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
