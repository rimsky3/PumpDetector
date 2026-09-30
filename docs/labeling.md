# Labeling

The target variable is created by hand. A trader marks pump intervals on the charts, and every bar inside
a marked interval becomes a positive example. Boundaries are moments in time, not bars: they can be set on
1h, 15m or 5m charts, whichever shows the start most clearly.

## What counts as a pump

A pump is the **impulse phase**: the few hours when open interest and price accelerate together, up to the
collapse of OI. Not the whole accumulation before it and not the drift after it. Since 2026-09-30 all tickers are
labeled this way on 5m/15m charts (median pump length about 4 hours, before that about 13 hours of hourly bars).

- **Start:** the bar where OI and price start accelerating together, set on 5m or 15m where it is visible.
- **End:** the sharp drop in open interest after the price peak; the collapse bar is the last bar.
- **Pump (`pump`):** the impulse develops, with strong growth in both OI and price (typically tens to
  hundreds of percent). Two impulses separated by a pullback are two pumps.
- **False start (`false_start`):** looks like the beginning of a pump (OI and/or price jump) but does not
  develop. These are valuable negative examples: they teach the model not to react to every spike.
- **Other (`other`):** anything worth recording that fits neither type.

For training, `pump` = 1 and everything else = 0. The `false_start` label is kept so that errors on
false starts can be analysed separately.

## Guidelines

- **Consistency matters more than precise boundaries.** Similar situations must get the same label.
- **Do not overlap intervals** on the same ticker. Two waves are two separate pumps; the gap between them
  stays 0.
- **Label ambiguous cases as `false_start` rather than skipping them.** Borderline examples carry the most
  information.
- **Label with hindsight, as a trader who knows the outcome.** Whether the pump was recognisable in
  advance is the model's job to find out.
- **Do not label listing artefacts.** A step in summed OI when a new exchange lists the coin is marked
  on the chart with a dashed line and is not a pump.
- **Set the start where it is visible.** If 15m or 5m shows the pump starting at 13:05, put it at 13:05;
  there is no need to round to the hour or re-mark it on another timeframe. An hour-rounded start teaches a
  15m model that the first 30–50 minutes of a pump are a normal market. Mixed precision across pumps is fine.
- **Use the notes column** when something other than OI drove the decision (funding, liquidations,
  news). It helps compare the model's feature importance with trader intuition later.

## `pumps.csv`

`pumps.csv` is the single source of truth for labels. The `pumps` table in the database is rebuilt from it.

```csv
base_asset,start_ts,end_ts,pattern_type,notes
AKE,2026-09-18 13:05,2026-09-20 10:00,pump,
G,2026-09-08 05:00,2026-09-08 13:00,false_start,OI up 30% without follow-through
```

| Column | Meaning |
|---|---|
| `base_asset` | Base asset, as in `tickers.txt` (`AKE`, not `AKEUSDT`). A label applies to all exchanges. |
| `start_ts` | Moment the pump starts, UTC, `YYYY-MM-DD HH:MM` (any minute) |
| `end_ts` | Moment the pump ends, **exclusive**: the close of its last bar |
| `pattern_type` | `pump`, `false_start` or `other` |
| `notes` | Free text, optional |

The database stores the same values. **Which bars are labeled** is decided by one rule for every timeframe
(`features.in_pump`): a bar with open time `t` and length `bar` is in the pump when the pump is already under
way at the bar's close, `start_ts < t + bar <= end_ts`. The model scores a bar at its close, so the label
answers the same question. With a start at 13:05: the 5m bar 13:05 is the first positive, the 15m bar 13:00
is positive (the pump has run 10 minutes by 13:15), and so is the 1h bar 13:00. For boundaries on whole hours
and hourly bars this is the familiar `start_ts <= t < end_ts`.

Until 2026-09-29 `end_ts` in the CSV meant "last hourly bar, inclusive"; those rows were converted (+1 hour),
and the labels did not change (checked on all 13,210 hourly bars).

`load_pumps.py` rejects the file if a row has an unknown type, ends before it starts, or overlaps another
interval of the same ticker.

## Labeling tool

```bash
python label.py
```

This opens http://127.0.0.1:8765, a local page served only to this computer.

- Synced panels drawn with TradingView Lightweight Charts (the same local copy as the dashboard, `app/vendor/`):
  candles (Binance), open interest (sum across exchanges, USD), volume (USD). Pane borders can be dragged.
- **Timeframe 1h / 15m / 5m.** 1h is Coinalyze (all exchanges); 15m and 5m come from Binance and Bybit directly
  (`python fetch_exchange.py`), so sums cover those two exchanges only, and Binance OI goes back 30 days.
  Switching keeps the selection and the visible time range. The model pane and error review are hourly and
  show on 1h.
- The line above the chart shows the hovered hour: OHLC, 24-hour change of price and OI, OI per exchange, volume.
- Mouse wheel zooms, drag pans, and the 7 / 14 / 30 / 60-day buttons set the window. **Лог. шкала** (log scale) helps with large moves.
- Click the first candle of an interval (start = its open), then the last (end = its close). The two clicks may
  be on different timeframes. Pick the type, add a note, press **Додати в список** (add to list). The UI is in Ukrainian.
- **Ruler** (button or Shift + click): measure a span — price change, bars, volume, OI change in USD and coins,
  model probability. Handy for judging whether an OI spike is big enough to label. Ruler clicks never set
  interval boundaries.
- Each change is written to `pumps.csv` immediately and synced to the database.
- Labeled intervals stay on the chart, numbered; each ticker button shows how many intervals it has.

**Ticker panel (right).** Lists the tickers in the database, all Binance USDT perpetuals (with 24 h change and
volume), or the training set; search, sort, ↑/↓ to step through. To label a new coin, find it under
**Усі Binance** and press **Завантажити** (load): the tool fetches 1h from Coinalyze and then 15m/5m from Binance
and Bybit in a background queue (one ticker at a time, since the Coinalyze limit is shared with the screener),
and opens the ticker when done. No files to edit.

**What goes into training.** Every labeled interval of every ticker does. The *unlabeled* hours of a ticker
count as "no pump" only when the ticker is in `tickers.txt`, which means "I reviewed the whole ticker". The
**У навчання** button adds the open ticker there (use it after looking through all 60 days, including a coin with
no pumps at all, which is a useful negative example). Loading or labeling a single interval never adds it. This
matters for the journal: confirming one signal on a watchlist coin as `pump` or `false_start` adds just that
interval to training, without declaring the rest of the coin's history pump-free.
`http://127.0.0.1:8765/#AKE` opens a specific ticker directly.

## Reviewing model errors

When `reports/oof_predictions.csv` exists (written by `python train.py`), the tool adds:

- **A fourth panel, "Модель"**, with the pump probability, the signal threshold (dashed) and a strip
  classifying each hour: signal inside a pump, false signal, signal on a false start, missed pump hour.
- **A model selector** (LightGBM, logistic regression, rule).
- **An error list for the ticker**: false-signal episodes, signals on false starts and missed pumps, each
  with a **Показати** button that zooms the chart to it. Below it, the delay of the first signal for every
  caught pump.

Predictions are out-of-fold: each ticker is scored by a model that never saw it, so this is how the model
would have behaved on that coin live. Errors are computed against the **current** `pumps.csv`, so a label
fixed in the tool immediately changes the error list. Retrain to see the model's reaction.

**Change a label only when the labeling rules justify it, never because the model disagrees.** The model
points at places worth a second look. The decision must come from the rules above (for example, "OI
started rising steadily three hours earlier"). Relabeling towards the model's output makes the next
model learn its own predictions: metrics improve while real quality does not.

To move an interval, delete it in the list and mark it again; overlapping intervals are rejected.

Things to look for:

- **Confident signals on a false start** (probability near 1). Either the model is fooled by something
  specific, or the interval is actually a pump.
- **False signals right before or after a labeled pump.** Red hours at the edges of a caught pump usually
  mean the boundary could move, not that the model is wrong.
- **Missed pumps with low probability.** Look at which features were unusual. The pump may be a
  different pattern that needs more examples.
