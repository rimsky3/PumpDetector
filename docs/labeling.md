# Labeling

The target variable is created by hand. A trader marks pump intervals on the charts, and every hourly
bar inside a marked interval becomes a positive example.

## What counts as a pump

- **Start:** open interest begins to rise steadily above its normal level. A long preparatory phase is
  not required.
- **End:** the sharp drop in open interest after the price peak. The bar of the collapse is the last bar.
- **Pump (`pump`):** the move develops, with strong growth in both OI and price (typically tens to
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
- **Use the notes column** when something other than OI drove the decision (funding, liquidations,
  news). It helps compare the model's feature importance with trader intuition later.

## `pumps.csv`

`pumps.csv` is the single source of truth for labels. The `pumps` table in the database is rebuilt from it.

```csv
base_asset,start_ts,end_ts,pattern_type,notes
AKE,2026-09-18 13:00,2026-09-20 09:00,pump,
G,2026-09-08 05:00,2026-09-08 12:00,false_start,OI up 30% without follow-through
```

| Column | Meaning |
|---|---|
| `base_asset` | Base asset, as in `tickers.txt` (`AKE`, not `AKEUSDT`). A label applies to all exchanges. |
| `start_ts` | First bar of the interval, UTC, `YYYY-MM-DD HH:MM`, bar open time |
| `end_ts` | **Last bar of the interval, inclusive** |
| `pattern_type` | `pump`, `false_start` or `other` |
| `notes` | Free text, optional |

In the database, `end_ts` is stored as an **exclusive** bound (`end_ts + 1 hour`), so a one-bar interval
still satisfies `end_ts > start_ts`. A bar with open time `t` is labeled when
`start_ts <= t < end_ts` (database form).

`load_pumps.py` rejects the file if a row has an unknown type, ends before it starts, or overlaps another
interval of the same ticker.

## Labeling tool

```bash
python label.py
```

This opens http://127.0.0.1:8765, a local page served only to this computer.

- Synced panels drawn with TradingView Lightweight Charts (the same local copy as the dashboard, `app/vendor/`):
  candles (Binance), open interest (sum across exchanges, USD), volume (USD). Pane borders can be dragged.
- The line above the chart shows the hovered hour: OHLC, 24-hour change of price and OI, OI per exchange, volume.
- Mouse wheel zooms, drag pans, and the 7 / 14 / 30 / 60-day buttons set the window. **Лог. шкала** (log scale) helps with large moves.
- Click the first candle of an interval, then the last. Pick the type, add a note, press **Додати в список** (add to list). The UI is in Ukrainian.
- Each change is written to `pumps.csv` immediately and synced to the database.
- Labeled intervals stay on the chart, numbered; each ticker button shows how many intervals it has.

To label a new ticker: add it to `tickers.txt`, run `python fetch.py --new`, restart `label.py`.
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
