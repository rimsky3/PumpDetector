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

**Run the 15m model on a ticker.** The 15m model scores (automatically when a ticker is opened; **Перерахувати модель 15m** re-runs it) the ticker's whole 15m history with
the 15m model and the hourly LightGBM on the fly (nothing is written to `signals`, so the live journal stays
clean). The model pane shows the 15m probability with Alerts in red and Candidates in amber on 15m and 5m charts
(on 1h, the hourly maximum, unless the ticker has hourly out-of-fold predictions). The same signals are drawn on the price pane: ▼ "Alert 0.98" / "Кандидат" above the bar where each episode starts, and a thin red/amber strip at the bottom of the price pane on every signal bar, so they read against your purple labeled zones. A card lists every signal
episode against your labels and sums it up: pumps caught by Alert and by Candidate, delay, false starts hit,
false episodes per month; it is recomputed after every label change. If the ticker was in training, the card says
so: that is not a test. The intended loop: pick a coin the model has not seen, run the model, label the whole ticker
by your rules (not only the model's mistakes: labeling only errors would skew training), then press **У навчання**.
With fewer than 30 days of 15m history, **Дозавантажити 60 днів** fetches the rest.

**Right panel tabs.** *Tickers* (list, search, loading), *<TICKER> · trades* (quick view of the open ticker: trades at Alert, signal summary, your stops across all tickers; ↑/↓ still switches tickers) and *Model* (status, retrain, candidate). The selected tab is remembered in the browser.

**Model status and retraining.** The *Model* tab shows which model is live (date, tickers, pumps) and, on each
ticker, whether it is **у моделі** (the live model was trained on exactly this labeling), **змінено** (labels or
"У навчання" changed since) or **не в моделі** (will enter the next training). Label fingerprints are stored in
the model file from 2026-09-30 on. **Перенавчити** runs `features.py` and `train.py --candidate` in the background
(~7–8 minutes); the new model is only a candidate until you compare it with the live one and press **Прийняти**
(the live model is backed up) or **Відхилити**. Restart `screener.py` and `dashboard.py` after accepting.

**Comparison page** (`http://127.0.0.1:8765/compare`, link *Детальне порівняння* in the panel): summary cards (new
value, old value, change marked better/worse) for the Alert or Candidate tier; a **fair comparison** block over tickers
where both models' scores are out-of-sample (tickers the live model never saw, or its saved out-of-fold predictions);
each model's own validation; a per-ticker chart and table (hatched = the live model saw that ticker in training,
not a fair comparison); and every labeled interval with filters *caught only by the new / only by the live / both /
neither / false starts*. Models trained from 2026-09-30 save their out-of-fold 15m predictions (`oof_15m.parquet`)
next to the model file, so once the live model has them the whole comparison becomes fair.

**Model on the chart (M).** The **Модель на графіку** button (or **M**) hides the model's signals, probability pane
and P value, leaving only your labels. Label a new ticker with the model hidden, then show it and compare: labels
made while looking at the model's Alerts drift towards them (anchoring), and the model would then partly confirm
itself.

**Editing a labeled pump on the chart.** Click inside a labeled interval to select it (it turns orange) and an
edit panel opens under the chart: type, note, stop, delete. Drag the handles on the chart: the left and right edges
move the start and end (snapping to bars of the current timeframe), the horizontal red handle moves the stop.
**S** then a click sets the stop (entry = price at the start). Esc or a click outside closes the editor.

**My entry and stop.** While selecting a pump (after the start and end clicks), press **S** (or **Стоп (S)** in the
selection panel) and click the stop level on the chart. The entry defaults to the price at the start of the impulse;
press **E** and click to set a different entry bar and price. Both are saved with **Додати в список**. For pumps
already in the list, the **Стоп** button in the table takes one click: the stop level (entry = price at the pump's start).
They are stored in `pumps.csv` as optional `entry_ts`, `entry_price`, `stop_price`. From each, the tool computes your
stop in the model's units, k = (entry − stop) / ATR(15m) at the entry bar, and the model card shows the median and
spread of your k against the model's stop. With 15–20 of them you can compare your practice with the data-derived
stop, or adopt it. Esc cancels, × clears.

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
