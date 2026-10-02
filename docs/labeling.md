# Labeling

The target variable is created by hand. A trader marks pump intervals on the charts, and every bar inside
a marked interval becomes a positive example. Boundaries are moments in time, not bars: they can be set on
1h, 15m or 5m charts, whichever shows the start most clearly.

## What counts as a pump

A pump (an *Impulse* in the interface) is the **whole move the owner would hold as one trade**: from the moment
open interest and price start accelerating together to the collapse of open interest. Not the accumulation before it
and not the drift after it. Its length is whatever the market gives: one to three hours for a short impulse, ten
hours for a long one that pauses on the way. Labeled on 5m/15m charts since 2026-09-30; the end rule was made
explicit on 2026-10-01.

- **Start:** the bar where OI and price start accelerating together, set on 5m or 15m where it is visible.
- **End:** the reversal from the peak has become clear — a 15m close below *peak − ⅓ of the move* (move = peak
  minus the start price; the peak is the highest high so far) **and no new high within about 3 hours after it**,
  **or** open interest down 10%+ from its peak, whichever comes first. The rule applies once the move is ≥ 5% (a third of a tiny move is
  noise). Use closes, not wicks. It is a guide: a clear reversal earlier (OI collapse, a flush) ends it earlier.
- **Pause, not an end:** a pullback after which the move makes a new high (even a deep one) does not split the
  impulse — it stays one interval. Checked on the owner's 64 impulses: the rule's end is a median 35 minutes from the
  labeled end (95 minutes without the 3-hour pause check).
- **Why a third:** on 63 labeled impulses, at the labeled end price had given back a median 32% of the move
  (22–42%) while open interest was only 6% below its peak — the owner exits on price; OI collapses later.
- **Pump (`pump`):** the impulse develops, with strong growth in both OI and price (typically tens to
  hundreds of percent).
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
- **Профіль** (volume profile): **День / Тиж** — a profile per UTC day or week; **Фрагмент** or **V** — a fixed-range
  profile (click start, click end; drag edges; × or Del removes). Same tool as in the dashboard; its clicks never set
  interval boundaries either.
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

**Blind signal review** (`http://127.0.0.1:8765/review`, link *Сліпий перегляд сигналів* in the header; logic in
`review.py`). Meta-labeling data: the model's Alert trades on coins outside training (backtest runs plus the screener
journal, closed trades only, shuffled with a fixed seed) are shown one by one with the chart **cut at the close of the
signal bar** — the server never sends later bars, the ticker and date are hidden (time is shown relative to the entry)
until you press *показати тікер і дату*, and the trade result is not shown. For each signal: *Зайшов би* (T) /
*Пропустив* (S) / *Не можу оцінити* (U), setup (B/P/O), reasons (tags, keys 1–9), an optional stop (click the price
chart) and a note; Enter saves and opens the next one, → postpones, ← goes back. Decisions go to `reviews.csv` (source
of truth, like `pumps.csv`). **Результати** compares your "would take" trades with all signals — same exit rules, and
with your own stop where you set one — and breaks them down by setup and reason. Look at it rarely (every 50–100
decisions): seeing results changes later decisions. Missing 5m data for backtest coins is fetched from the exchanges
on demand (only the ~2.5 days the chart shows, a few seconds; the next five signals are prefetched in the background). `python review.py --rebuild` refreshes the pool
after new backtest runs; `--summary` prints the same summary.

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

**Terminology in the interface.** The tools call a labeled `pump` an *Impulse* and a `false_start` a *Failed
impulse*: the model detects OI-driven impulse moves, of which pump-and-dumps are one kind. The values stored in
`pumps.csv` and the database are unchanged.

**Setup (optional, since 2026-10-01).** Each impulse can carry a `setup` column in `pumps.csv`:
`breakout` — the move leaves a range (price sideways for hours, OI flat, then price and OI accelerate together);
`pullback` — a new impulse after an earlier move has ended (it gave back a third, or OI reset by 10%+) while price kept
at least half of that move's gain and then accelerated again (a pause with OI holding is part of the same impulse, not a new interval); `other` — anything else or unsure. The model
does not train on it — the target is the same impulse — it only lets metrics be split by setup. In `label.py`: the
*Сетап* select in the new-interval form, the edit bar and the table; keys **B / P / O** on a selected interval;
**Підказати сетапи** suggests a setup for impulses without one (a move of ≥ 8% in the 24 h before the start or a
labeled impulse that ended ≤ 24 h earlier, a pullback from that high, and price still holding ≥ 50% of that gain →
pullback; otherwise breakout), with the reason on hover; you accept each one or all at once.

**Model-assisted labeling (since 2026-10-01).** The model never labels on its own — only what you confirm reaches
training (self-labeling would teach the model its own mistakes). Two helpers in `label.py`:
- **Чернетки імпульсів** (toolbar): for the open ticker, drafts from Alert episodes that are not labeled yet. Start =
  the price low in the 6 h before the first signal; end = the labeling rule (a close below peak − ⅓ of the move, or OI −10%; if
  neither has happened yet, the move is still running and the end is the last bar). *Показати* puts a draft as the selection to
  adjust and add; *Прийняти* adds it as is; *Відкинути* hides it (remembered in the browser). All drafts are shaded blue on
  the chart with a "чернетка N" mark; a bar right under the chart steps through them (**N** next, **Shift+N** previous)
  with *Прийняти* (takes the current, possibly adjusted, bounds) and *Відкинути*. Pressing the button again hides them. Open drafts only after
  you have looked at the chart yourself, to avoid anchoring.
- **Черга** (right panel tab): where labels help most (active learning) — screener tickers with unlabeled signals,
  scored Alert ×3, Candidate without hourly confirmation ×2, near-miss (15m at 0.85–1 of its threshold) ×1, +2 for
  tickers outside training; for tickers in `tickers.txt` only the last 7 days count. Important there: unlabeled hours
  of a ticker in `tickers.txt` train as "no impulse".

**Out of scope.** A coin that cannot be traded profitably however clean its setups look (jerky price, manipulative
flushes both ways, thin order book) is excluded with **Виключити** in `label.py`. It goes to `excluded.txt` with the
reason and is dropped from training (even its labeled intervals), from `tickers.txt`, the market scanner, the screener
and the dashboard list. First one: US.

**Dragging bounds.** Any selection — a new one or a draft put up with *Показати* / **N** — and any saved interval picked
on the chart has two orange edge handles: drag them like a window edge (↔ cursor) to move the start or the end. A
selection is saved with *Додати в список* or *Прийняти*; a saved interval is saved on release.
