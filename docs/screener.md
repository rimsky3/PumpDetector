# Screener and dashboard

The live product: a screener that scores a watchlist every 15 minutes and a dashboard with two screens, Forecast
and Journal, built from the "Pump Detector App" design mockup.

**Two signal tiers** (since 2026-09-30, see [modeling.md](modeling.md#experiment-two-tier-signal-15m-candidate-hourly-confirmation)):

| Tier | Rule | Meaning |
|---|---|---|
| **Candidate** | 15m model ≥ its threshold | earliest (~30 min after the start), noisy: reacts to ~⅔ of false starts. Check the order book. |
| **Alert** | Candidate and the hourly LightGBM ≥ 0.75 × its threshold (last closed hour) | same pumps caught, false starts and false alarms as low as the hourly model |

```
train.py ──saves──▶ models/logreg.joblib, models/lightgbm.joblib        hourly models
                    models/lightgbm_15m.joblib                          15m model + confirm rule + 15m stop

screener.py, every 15 minutes (HH:02/17/32/47 UTC):
    15m bars from Binance/Bybit + 15m liquidations from Coinalyze → features.build_15m() → 15m model → signals
    once an hour, also: Coinalyze 1h for the last 10 days → features.build() → hourly models → signals

tiers.py (at read time): 15m score + last closed hourly score → Candidate / Alert

dashboard.py + app/dashboard.html:  signals + raw series + tiers + risk  →  dashboard
```

## Running it

```bash
python train.py                        # trains, validates and saves models/ (needed once, then after changes)
python screener.py                     # one cycle
python screener.py --loop              # keep running: 15m cycle every 15 minutes, hourly models once an hour
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

## Market scanner

`scanner.py` fills the list automatically. Every 15-minute screener cycle it takes all Binance USDT perps (one request)
and, for those with at least 5 M$ of 24h turnover (~240 of ~530), fetches 7.5 days of closed 15m candles and 4 hours
of open interest (Binance public API, no Coinalyze calls; ~50 s). A coin added mid-hour gets its hourly Coinalyze
data in the same cycle, so Alert (which needs the hourly model) works right away. A coin enters the auto-list when

- **liquidity**: 24h turnover ≥ `min_turnover` and open interest ≤ `max_oi` (large caps are out of scope), and
- **waking up**: the last 4 hours of volume are ≥ `vol_x` × the 7-day median of 4-hour volume while price is not
  falling, **or** open interest is up ≥ `oi_pct` % and price ≥ `price_pct` % over 4 hours.

**Cycle order (latency).** Each 15-minute cycle first scores the list (15m data → 15m model), then onboards new
tickers, then — once an hour — collects hourly Coinalyze data, starting with tickers whose 15m model is above its
threshold so that their hourly confirmation (needed for Alert) is ready in ~1–2 minutes instead of after the whole
list (~23 calls per ticker at 40 per minute). The market scan runs last; coins it adds are onboarded at once
(hourly data, both models). A Candidate therefore appears ~3–5 minutes after the 15m bar closes.

**Training tickers outside the list.** Every 3 hours the screener also refreshes 15m/5m exchange data (Binance, Bybit,
OKX) and 15m liquidations for `tickers.txt` coins that are not in the list. They are not scored; the point is history
that the sources do not keep for long (OKX open interest ~14 days, Coinalyze 15m liquidations 1–2 months), so that
OKX can later be tested as a model input.

The filter is deliberately looser than the model: its job is not to miss a pump; the model decides whether it is one.
An auto coin stays `ttl_hours` (48) after its last trigger or model signal, at most `max_auto` (30) coins. Alerts,
not list entries, trigger notifications.

Presets on the 56 labeled pumps (`python scanner.py --recall`, hourly data from the database, same 4-hour windows; "on time" = the coin
would have been in the list before the impulse started):

| Preset | Turnover ≥ | Volume × / OI % / price % | On time | Only during | Never |
|---|---|---|---|---|---|
| Strict | 30 M$ | 4 / 15 / 3 | 24 | 20 | 12 |
| Balance | 20 M$ | 3 / 10 / 2 | 30 | 17 | 9 |
| Wide | 5 M$ | 2 / 6 / 1 | 47 | 9 | 0 |

The turnover floor costs most: many targets are quiet small coins until the pump starts (10 M$ with the Balance
triggers gives 37/56). At the first live scan the presets passed 3 / 5 / 9 coins.

The dashboard's **Сканер** card previews any preset against the last scan before saving, lists the auto coins,
and adds a coin by hand (**＋ у watchlist**: validated against Binance, written to `watchlist.txt`, data fetched at once).

**Ticker context in the dashboard.** The metrics card shows which share of the ticker's open interest the model sees
(30-day average over all exchanges from Coinalyze; the 15m model uses Binance and Bybit only) as a bar per exchange,
amber below 70%. A card under the position summarises the model's closed trades on this ticker over everything it
has scored (including the history scored when the ticker joined the list), flagged when the ticker is in training.

**History versus live on the chart.** Trade marks from scores computed after the fact (history scored when a coin
joins the list) are dimmed and tagged "іст."; live ones are bright. For auto coins a blue "сканер" mark shows when the
scanner, with the current thresholds, would have added the coin had it been running all along
(`scanner.entries_on_history`, the same rule vectorised over Binance 15m data).

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
  4 h, volume z-score, share of long accounts; the model's simulated position with stop and exit
  state;
- *centre*: chart ([TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts),
  Apache 2.0, local copy in `app/vendor/`): candles with signal zones, entry and exit markers of every
  simulated trade, the stop line, OI candles (in coins by default, USD on a toggle), volume and the
  hourly probability with the threshold;
- *right*: the watchlist like a trading terminal — search, **Refresh**, filters (all / Alert /
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
| Exit | probability below the threshold for 2 bars in a row, or the stop is hit |

**MAE** (maximum adverse excursion) of a pump is how far the price fell below the entry before reaching the
peak. Entry is the close of the first labeled bar. The fall after the peak is not included: that is the
exit, not a pullback the stop had to survive. On 18 pumps: k = 2.9, median MAE 1.1 ATR, median move to the
peak 18.8 ATR.

A "trade" on the dashboard starts at the first hour with probability ≥ threshold and ends by the exit rule.
It is a simulation to judge the model, not trading advice.

## 15m tier details

- **Data per 15m cycle:** a new watchlist coin gets 10 days of 15m history; the rest get exactly the gap since their
  last stored bar plus a margin (at least 6 hours, at most 29 days), so a laptop that was off overnight is back-filled
  without holes. The hourly Coinalyze part does the same (10 days normally, the whole gap after a long pause, up to 60).
  Data that exchanges keep (candles for years, OI 14–30+ days) can be back-filled later; live signals, the live
  forward test and data without history (order book, Hyperliquid OI) need the screener running, i.e. a server. Funding
  is always fetched with 2 days of lookback so the last paid rate is known.
- **Coinalyze budget:** hourly collection ~15 calls per ticker once an hour, 15m liquidations ~3 per ticker every
  15 minutes: about 300 calls an hour for 13 tickers out of 2,400.
- **Hourly context** for a 15m bar is the score of the last hour closed by the bar's close (13:30–13:45 → the 12:00
  hour), so nothing from the future. Tiers are computed when read, so the order of the 15m and hourly parts
  inside one cycle does not matter.
- **Risk on 15m:** stop = entry − k·ATR(14 bars of 15m), k from 15m MAE of labeled pumps (~1.2); no trailing stop (removed 2026-10-01: the owner exits by the order book);
  exit when the Alert is gone for 4 bars (1 hour).
- **Chart timeframe 5m / 15m / 1h** is independent of the model: on a coarser chart a bar shows the highest score
  and tier inside it (an Alert within the hour is not lost), on a finer chart the last score known at the bar's
  close. Positions are computed on the model's own bars; entry/exit marks go to the chart bar that closes with them.
  The screener refreshes a 5m tail for this (no liquidations: the model does not use 5m).
- **Live candle.** The page connects straight to the exchanges from the browser: Binance
  `wss://fstream.binance.com/market/ws/<symbol>@kline_<tf>` (the old `/ws/` path accepts connections but sends
  nothing), Bybit `kline` and `tickers` topics, and Binance open interest over REST every 5 s. On 5m/15m charts the
  forming bar shows price, volume (Binance + Bybit) and OI in coins; on 1h only price, because hourly volume and OI
  come from Coinalyze across all exchanges. The model is not re-scored live: it only scores closed bars. When the
  chart is opened and the database lacks the latest closed bars, the server fetches the tail in the background and
  the page reloads about 15 s later.
- **Journal:** `15m + 1h` rows are Alert trades, `15m кандидат` rows are Candidate trades, both on 15m bars.
  Hourly LightGBM and logistic regression stay selectable for comparison.

## Next steps

- Coins outside the training set in `watchlist.txt`, so live scores are genuine out-of-sample.
- Forward-test report from `live` rows once a few weeks accumulate: signals vs what happened.
- Telegram notifications for new signals (the mockup's signal log already has room for them).
- Deployment to the VPS with Docker Compose: database, screener loop and dashboard as services.
