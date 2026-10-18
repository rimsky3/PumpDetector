# Screener and dashboard

The live product: a screener that scores a watchlist every 15 minutes (the "hot" coins every 5) and a dashboard with two screens, Forecast
and Journal, built from the "Pump Detector App" design mockup.

**Two signal tiers** (since 2026-09-30, see [modeling.md](modeling.md#experiment-two-tier-signal-15m-candidate-hourly-confirmation)):

| Tier | Rule | Meaning |
|---|---|---|
| **Candidate** | 15m model ≥ its threshold | earliest (~30 min after the start), noisy: reacts to ~⅔ of false starts. Check the order book. |
| **Alert** | Candidate and the hourly LightGBM ≥ 0.75 × its threshold (last closed hour) | same pumps caught, false starts and false alarms as low as the hourly model |

```
train.py ──saves──▶ models/logreg.joblib, models/lightgbm.joblib        hourly models
                    models/lightgbm_15m.joblib                          15m model + confirm rule + 15m stop
                    models/lightgbm_15m_start.joblib                    start model (first 30 min of an impulse): entry of the split mode

screener.py, every 15 minutes (HH:02/17/32/47 UTC):
    15m bars from Binance/Bybit + 15m liquidations from Coinalyze → features.build_15m() → 15m model (+ start model,
    if present: signals.model = lightgbm_15m_start; history for a new one — screener.py --backfill-15m DAYS) → signals
    once an hour, also: hourly model from our own 15m data (seconds) → signals; Coinalyze 1h for the coins that need it
      → hourly models → signals — in a BACKGROUND thread (see below)
    in between, every 5 minutes (:07, :12, :22, …): 5m tail for the whole list → "waking up" (wake.py) →
    "hot" coins (15m score ≥ 0.8 × threshold in the last 2 h, or woke up in the last 2 h):
    5m bars + 5m liquidations → rolling 15m bar that just closed (offset 5 or 10 min) → same 15m model
    → signals as model lightgbm_15m_roll

background thread (--loop): the slow Coinalyze downloads never hold up the 15m and 5m cycles —
    the hourly Coinalyze part, the 1h history of coins the scanner just added (they get exchange contracts with two
    calls, 15m data, a 15m score and an hourly score from own 15m right away), the 3-hourly history of training coins
    outside the list. Background Coinalyze calls have lower priority (fetch.Coinalyze.background: they leave 10 of the
    40 calls/min and wait while urgent 15m / 5m calls queue). On 2026-10-03 the hourly part took 27 minutes and AIN's
    first 15m bar was scored 9 minutes late; now the main cycle returns in ~25 s.

tiers.py (at read time): 15m score + last closed hourly score → Candidate / Alert

dashboard.py + app/dashboard.html:  signals + raw series + tiers + risk  →  dashboard
```

## Running it

```bash
python train.py                        # trains, validates and saves models/ (needed once, then after changes)
python screener.py                     # one cycle
python screener.py --loop              # keep running: 15m cycle every 15 minutes, hot coins every 5, hourly once an hour
python screener.py --no-fetch          # score data already in the database, no API calls
python dashboard.py                    # dashboard at http://127.0.0.1:8766
```

## Rolling 15m bars (every 5 minutes)

The 15m model only sees a bar after it closes, so a move that starts at :01 waits until :15 (+2 min for the
exchanges). Between the regular cycles the screener rescores the coins that are close to a signal on a **rolling**
15m bar: 5m bars and 5m liquidations are rolled into the 15 minutes that just ended (e.g. 16:25–16:40). The model is
unchanged. On the training tickers this gave Alerts 5 minutes earlier (median 35 → 30 min after the labeled start)
and Candidates 3 minutes earlier, with no extra false alarms (4.2 → 4.1 per 30 days, `experiments/early_5m.py`).

- **Hot coin**: the 15m score was ≥ 0.8 × threshold at some point in the last 2 hours. A quick cycle covers only these
  coins and takes about 15–25 s. Coins with too little 5m history get 9 days of it once, because the features need a
  week of data.
- A coin is scored only when **every** market already has the last 5m bar for both candles and OI. Exchanges publish
  the OI for a 5m bar after it closes, and a half-built bar looks like an OI crash to the model.
- Rolling scores are stored as a separate model, `lightgbm_15m_roll`. The **watchlist** shows the newest score
  (status and P), and **notifications** start an episode from it. The journal, the position card, chart marks and the
  backtest stay on the regular 15m bars, the same bars the model was trained and validated on. When the newest rolling
  score is more recent than the last scored regular bar, the chart shows it as a separate point on the P pane (axis
  label "5m", colored by its level) and the info row adds "P 5m", so a notification never disagrees with the chart.
  Candidates and Alerts that existed only on rolling windows stay on the chart's history too: a faint zone on the price
  pane and a "5m 0.90" dot above the P column of the bar that contains the window's end (VIRTUAL 2026-10-04 16:55 was an
  Alert only on the 16:55–17:10 window; both regular bars around it scored 0.80).
- 5m liquidations for hot coins are always fetched for 10 days. Coinalyze counts calls per symbol, not per day, so this
  costs the same as a short tail and fills any gap from the time the coin was "cold".

## Modes (since 2026-10-02)

The left panel's "Режим" dropdown replaces the model buttons: a mode is a model plus trade rules (`strategies.json`):
"15m + 1h · market" (main), "· limit −0.5 ATR", "· OB ≥ 1", "· OB + limit", "15m · entry on Candidate", and the hourly
models for comparison. Signals are the same for all 15m modes — the screener runs once; a mode only changes which
signals become trades and how, so switching is instant. Each tab keeps its own mode (Forecast, Journal and Backtest are switched separately): Forecast — position, trade marks,
history card; Journal — which mode's trades are listed (the table still shows one row per mode); Backtest — the mode of
the next run (15m modes only). For past signals without an order book snapshot the
OB filter takes the signal. In Backtest the retrained model waiting for "Прийняти" is called "нова версія" to avoid
confusion with the Candidate signal tier.

## Mode "breakeven on p" (since 2026-10-07)

`strategies.json` mode `p_be`: when a footprint **p** mark appears in an open long, the stop moves to breakeven (the chart
line reads "Стоп · б/з (p)"); after a trade that saw p, no new entry until the Alert episode ends. In `experiments/p_exit.py`
(213 coins, 40 days) this was the best of seven variants — profit factor 0.94 vs 0.93, no large winner cut — but the gain
is within noise, so it runs as a separate journal row for a live comparison rather than replacing the main mode.

## Altcoin market state (since 2026-10-07)

`market.py` runs in its own screener thread every hour at HH:01 (and at start; the first run downloads 60 days): hourly
closes of ~250 liquid Binance alts and BTC → `market_breadth` — share of alts up over 7 days and 24 hours, median alt
move, alts vs BTC, BTC trend and volatility. **Hot** = more than 60% of alts up over 7 days, **Cold** = below 40%.
In the 6-month backtest Alerts in a Hot market were worse (profit factor 0.78 vs 1.09, every month), in the live journal
of 24.09–06.10 they were better (1.52 vs 1.04) — so it is shown as context and tested live: the journal has the mode
"15m + 1h · ринок не гарячий" (`market_cool`), which skips Alerts while the market is Hot. The live list (Binance) and
the experiment's list (Bybit cache) agree: correlation 0.997, same Hot / not-Hot state in 98.7% of hours.

Dashboard: the **Market ▾** button right of "OI монети / USD" shows the state (Hot / Neutral / Cold). Its panel: share of
alts up over 7 days with a Cold–Neutral–Hot gauge, a one-week chart (7-day share white, 24-hour grey; hover for the
hourly value), a one-line hint, alt and BTC metrics, and the backtest vs journal result. Refreshed every 5 minutes.

## Idiosyncratic movers (since 2026-10-07)

After each hourly market update, `movers.py` computes for the same ~250 alts: beta and R² against the equal-weighted alt
index (hourly OLS over 7 days, ending a day before now so the current spike does not bias it), total / systematic /
idiosyncratic return over 1, 4 and 24 hours, the residual z-score, and Amihud illiquidity relative to the coin's 30-day
norm → `reports/movers.json`. The dashboard shows it as the **Movers ▾** table (sortable, horizon switch, IN LIST /
add to watchlist). It describes the market, it is not a model signal.

Since 2026-10-08 the mark follows the coin outside the table: `movers.json` also keeps the metrics of all ~250 coins
(`all`), and `movers.info(base)` gives a chip for the pane header — `MOVER 4H · z +4.1 · idio +12%` (the horizon with the
largest |z| ≥ 2.5; red when the coin falls harder than the market; hover: beta, R², ILLIQ, turnover) — and a lilac
frame (`--mover`) around the watchlist row when the coin rises on its own, z ≥ 2.5 over 1H or 4H (24H moves get only the
chip, so the frame does not stay on all day). The threshold is the one behind the Movers button, so the button count and
the frames agree. Over two weeks (Sep–Oct) 5–7 of ~253 coins were above it at any moment.

## Drivers card (since 2026-10-08)

Under the probability card the dashboard shows **DRIVERS** for the last scored 15m bar (`drivers.py`): a stacked bar of
group shares (open interest, volume, price, own move, funding, long / short, other, and everything pulling down as
"Against"), then "Pushing up" — the four features that raised P most, with their current value and a bar for the
strength of the contribution — and "Pulling down" — the two that lowered it. Hovering a row explains the feature in
Ukrainian. `⚠ No volume behind this signal` appears when P is above the threshold while the volume group pulls it down
(a signal driven by positioning or correlation rather than a volume spike, e.g. CHIP 06.10 08:45). The owner chose this
layout (variant A with the warning from C) on the mockup https://claude.ai/artifact/Rf29hSf9fmsTynMqcT3yMj.

**DRIVERS chip** (since 2026-10-08): in the chart window header, next to status and P, the two groups with the largest
positive share of the drivers — e.g. `DRIVERS · OI 29% · Own move 25%`, coloured like the card's bar, ⚠ in front when the
signal has no volume behind it; hover for the top four groups and features. Shown only on Candidate / Alert and only when there is no MOVER chip (two chips wrapped the header to a second line);
with a MOVER chip the two groups go into its tooltip and the ⚠ in front of its text. So in practice it covers small coins
outside the Movers universe (avg turnover < 5 M$), where it is the only hint that a move is the coin's own.

## ALT-EW index for the 15m model (since 2026-10-08)

The 15m model sees the coin's move relative to the altcoin market (`beta`, `idio_*`, `illiq_x`, `park_*`; see
[features.md](features.md)). Its input is `alt_index` — the equal-weighted 5m return of the same ~250 alts as above.
`alt_index.watcher` updates it in its own thread 8 seconds after every 5m bar closes (~250 one-candle REST requests,
weight 1 each, ~13 s), well before the cycles at :02 / :07 / …; `score_15m` and the rolling scorer call
`alt_index.ensure` for the bar they score, so a late thread only delays, never zeroes, the features. The rolling scorer
loads 10 days of 5m data (beta and Amihud use a week that ended a day earlier).

## Liquidations after scoring (since 2026-10-07)

Models trained without liquidation features (the default from 2026-10-07) do not need them to score: the cycle takes
exchange data, scores the 15m bar and the hourly confirmation, and only then collects 15m liquidations from Coinalyze
in the background thread (`collect_liq_15m`, last 3 days; the 5-minute cycle collects 5m liquidations for hot tickers
after its rolling score). A model that still uses them (`uses_liq`) keeps the old order: liquidations first.

## Order book and limit hint (since 2026-10-02)

- **OB** — the bid / ask ratio within ±2% of the mid price, Binance + Bybit summed (`orderbook.py`, public order book
  APIs, no key). ≥ 1 means at least as many buyers as sellers in the book. On the backtest coins outside training, model
  entries with OB < 1 were the worst (profit factor 0.76 against 0.98; `experiments/book_spot.py`,
  `experiments/entry_rules_backtest.py`).
- The screener records a snapshot for every coin near a signal (15m score ≥ 0.8 × threshold in the last 2 hours) on each
  5-minute and 15-minute cycle into `book_ratio`. The chart shows "OB 1.38" above each Alert start (the first snapshot
  within 12 minutes after the signal bar closed; green ≥ 1, amber < 1); older Alerts have none.
- Display removed on 2026-10-02 (the left panel card, the "OB" chart marks and OB in notifications): the filter did not
  hold up over six months, and the owner reads the real order book in the terminal. Snapshots are still recorded for a
  later live re-test, and the OB modes stay in Journal and Backtest.
- **Limit hint**: a limit buy at the signal bar's close − 0.5 ATR, valid one hour — a tile "Ліміт" in the position card
  and a dashed "Ліміт" line on the chart while valid. On the backtest it was the only pullback entry that was not worse
  than a market buy in either half of the period. The position simulation still enters at market.
- Notifications add "OB 1.38 · ліміт 0.1092".

## Cross-exchange spread (since 2026-10-13)

During an impulse the same perpetual trades at different prices on different exchanges (DOS, 2026-10-12: at the peak
OKX was 5.5% below Binance; after the crash Bybit stayed 5% above Binance for 18+ minutes, against a normal 0.1–0.2%).
`spread.py` shows this and alerts on it. It is a "where to look" tool, not an arbitrage signal: depth, funding and fees
are the owner's check. Nothing is stored in the database (the owner's decision).

- **Monitor** — a thread in the dashboard process. Every 15 s it requests the best bid / ask of all perpetuals from
  Binance, Bybit and OKX (one request per exchange) and keeps three hours of mid prices in memory. Coins are matched
  through the `markets` table (so `PUMP` maps to `PUMPFUNUSDT` on Bybit) and by symbol name for the rest of the market.
- **Executable spread** of an exchange pair = bid of the expensive one / ask of the cheap one − 1: what a market order
  could actually capture, not the difference of last prices.
- **Event**: executable spread ≥ `min_pct` (1%) **and** ≥ `norm_x` (5) × the coin's normal spread for that pair (median
  absolute mid difference over `norm_hours`, floor 0.05%) **and** held for `hold_sec` (30 s). It ends after two
  snapshots below 0.7 × the threshold. A repeat on the same coin within `cooldown_min` is shown but not notified.
- **Safeguards**: a pair with a mid difference above `max_pct` (30%) is treated as two different assets under one
  ticker and ignored; a quote older than 3 minutes drops that exchange for the coin; no events until a pair has 20
  snapshots (about 5 minutes after a dashboard restart), so a permanent offset becomes "normal" instead of alerting.
- **Scope**: `list` — only coins of the dashboard list (default); `all` — the whole market above `min_turnover`.
- **Dashboard**: the "Δ" pane under the price (toolbar button Δ, per chart window) — the bar close on Bybit and OKX
  relative to the chart's exchange, from the per-exchange candles already in the database, with the last point live from
  the exchanges' WebSockets; the header chip "Δ Bybit +5.1%" (teal while an event is on) opens a card with bid / ask,
  top-of-book size, funding per exchange and the last three hours; a "Δ" glyph after the ticker in the list; the
  "Spread ▾" popover lists current and recent events and holds the settings form; "Sound ▾ → Spread" turns on the tone.
- **Telegram**: type "🔷 Spread" (docs/telegram.md).

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

Pruning the list (since 2026-10-07): a row whose coin had no Candidate / Alert and no "waking up" for 24 hours is
dimmed ("quiet N h" on hover, `quiet_h` in `/api/list`). The × on a row (on hover) removes the coin: a manual one from
`watchlist.txt`, a scanner one from the auto list, and both are blocked "until a new trigger"
(`reports/auto_snooze.json`, `scanner.block` / `release_blocks`): the scanner's windows (4-hour volume and OI) keep a spike
passing the filter for hours, so while the coin keeps passing it is the same event and stays out; the first scan where it
no longer passes (and at least `BLOCK_MIN_HOURS` = 1 h after removal, so a flickering filter cannot bring the same spike
back) lifts the block, and the next pass adds it again as an auto coin. "Undo" in the toast undoes it
(`/api/watchlist/undo`). The screener stops scoring the coin from its next cycle.

### Sections (since 2026-10-08)

The filter tabs (all / Alert / candidates / auto / ★) were replaced by four sections; every section collapses with its
arrow (state in `localStorage` `pd-sec`, QUIET collapsed by default), and ↑/↓ walk the open sections only. Search and the
★ "Favorites only" button filter across all sections. Labels in the list are in English.

| Section | What goes there |
|---|---|
| **NOW** — signal or event in the last hour | Alert / Candidate burning, or faded less than 1 h ago (`last_sig` = close of the last non-calm bar; 4 bars, like the position exit); Vol 5m (≤ 30 min); `b · Reclaim` (≤ 1 h after the bar); the first WOKE trigger (1 h). Several events — the latest one counts. |
| **WOKE UP** — dormant coins coming alive | auto coins of the WOKE channel (below); a two-line row with the trigger reason |
| **WATCHLIST** — liquid, scanner + manual | liquid-channel auto coins (tag **A**) and manual coins (`watchlist.txt`, no tag, no TTL) |
| **QUIET** — no signals for 24 h+ | dimmed rows, candidates for × |

Row chips: `Alert` / `Candidate` (dim with age once faded), `b · Reclaim · 10m`, `p · 25m` (orange; never lifts a row into
NOW), `Vol 5m ×6`, `Woke · 3h`. A manual coin that triggers WOKE stays in WATCHLIST with the `Woke` chip and spends its first
hour in NOW. A hot coin moves to NOW, it is never shown twice. `/api/list` rows carry `channel`, `wokeup`
(`{first, last, reason}`), `last_sig` and `fp` (latest live footprint mark).

## "Waking up" (5-minute early warning)

`wake.py` flags a coin from the list that has just started to move, before the model reacts: over the last 15 minutes
volume is ≥ 5× its norm (the median 15-minute volume of the previous 7 days) and ≥ 2× the median of the last 4 hours
(so a coin that is already moving does not "wake up" again — AIN 2026-10-03 18:05 was ×188 against its dead weekly norm
but half of its last 4 hours; in the backtest this drops 9% of events with the same quality), price is up ≥ 2%, and open interest
(Binance + Bybit, chained index) is up ≥ 2% over the last hour; then 4 hours of silence for that coin. It is not a model
signal and not an entry: in September backtests (86 coins ≥ 20 M$, outside training, `experiments/wakeup_5m.py`) a
Candidate followed within 2 hours in 46% of cases (6% after a random moment), and for 37% of Alerts the warning came
~30 minutes earlier; price afterwards had no edge either way.

- Every cycle (15-minute and 5-minute) the screener pulls the 5m tail for the whole list straight from the exchanges
  (~40 s for 40 coins, no Coinalyze calls) and checks the bar that just closed. Events go to `reports/wake_events.json`
  (7 days).
- A coin that woke up is **hot** for 2 hours: it gets 5m liquidations and a rolling 15m score every 5 minutes right
  away, instead of waiting for the regular 15m bar to cross the hot threshold.
- Dashboard: the candle and its volume bar are drawn violet with "×N" (volume vs norm) under the candle, on any
  chart timeframe (the bar that contains the 5m bar); the info row shows volume, price and OI change; in the ticker
  list a calm coin's dot is violet for 30 minutes after it woke up (hover for the numbers). Optional voice (the "Volume 5m"
  switch in the **Sound ▾** menu, see below): the browser says "Volume S T R K" (speech synthesis, ticker spelled) for each
  new event while the page is open; like the other sounds it needs one click on the page after a reload.
- `python wake.py AIN --days 3` lists a coin's events from the database.

## Footprint marks b / p (since 2026-10-07)

Right after the 15m scores of a full cycle the screener runs `footprint.detect`: for each list coin that had a Candidate
or an Alert within the last 6 hours (24 bars, including the bar just closed) it checks the bar that just closed for the
owner's "stop-run that gets bought back" pattern — **b**: the bar pierces the 2-hour low, ≥ 30% of the bar's volume
traded below that low, the bar closes back above it, ≤ 20% of the volume sits in the top third of the bar, volume ≥ 2×
the previous day's median; **p**: the mirror at the highs. Candles and volume come from the Binance 15m candles, the
footprint from `trade_bars` (the trade recorder's 5-minute footprints summed into the 15m bar — a bar is checked only if
all three 5-minute bars were recorded). Marks go to the `fp_marks` table; the dashboard shows them under (b) or above
(p) the candle with details on hover, the bot sends them (setting "b / p"), and the outcome of each mark (+2R before the
stop under the bar's low, stop, or 12 hours) is computed on read by `footprint.outcome`. Without the recorder running
(`trade_bars` missing or empty) nothing is marked. Why only in context and what the history says: docs/trades.md.

## Market scanner

`scanner.py` fills the list automatically. Every 15-minute screener cycle it takes all Binance USDT perps (one request)
and, for those with at least 5 M$ of average daily turnover (~300 of ~530), fetches 7.5 days of closed 15m candles and
4 hours of open interest (Binance public API, no Coinalyze calls; ~50 s). The 30-day average daily turnover comes from daily
candles, fetched once per UTC day for every perp (`reports/scan_turnover_avg.json`; the first scan of the day takes ~75 s). A coin added mid-hour gets its hourly Coinalyze
data in the same cycle, so Alert (which needs the hourly model) works right away. A coin enters the auto-list when

- **liquidity**: average daily turnover over the last 30 complete days ≥ `min_turnover` and open interest ≤ `max_oi`
  (large caps are out of scope). Until 2026-10-03 this was the 24 h turnover; backtest trades showed that a liquid coin
  often starts its move on a quiet day (SYN +51% with 7 M$ of 24 h turnover against a 36 M$ average), while small coins
  that just spiked above the floor do worse (`experiments/turnover_at_entry.py`). The 24 h turnover is still shown in the
  reason text. And
- **waking up**: the last 4 hours of volume are ≥ `vol_x` × the 7-day median of 4-hour volume while price is not
  falling, **or** open interest is up ≥ `oi_pct` % and price ≥ `price_pct` % over 4 hours.

**Hourly confirmation from our own data.** Since 2026-10-01 the hourly model is scored from our 15m exchange data rolled
up to hours (`features.build_hourly_own`) for every coin where Binance and Bybit hold at least 70% of open interest
(30-day average from Coinalyze) or that Coinalyze does not cover: the confirmation for the hour that just closed is
ready in the first seconds of the cycle instead of after the hourly Coinalyze collection. `experiments/hourly_from_15m.py`
and `rolling_confirm.py` showed the same decisions (98.6% of hours, same Alert events). Coins with more open interest
elsewhere (Hyperliquid, OKX) keep the Coinalyze path; hourly Coinalyze data for the others is refreshed every 6 hours
(1h chart, history).

**Cycle order (latency).** Each 15-minute cycle handles the **hot coins first** (since 2026-10-06): fresh 15m data and
15m liquidations for them only, the 15m score and — at the top of the hour — the hourly confirmation from our own 15m
data; then the same for the rest of the list. Before, every coin waited for the 15m liquidations of the whole list
(~200 Coinalyze calls at 40 per minute), so a bar was scored ~7–9 minutes after it closed and an Alert that needed the
new hour's confirmation appeared ~11 minutes after the close (RLC, 2026-10-06 10:45 bar → 11:11). Then new tickers are
onboarded, and once an hour the hourly Coinalyze part runs in the background, hot tickers first. The market scan runs
last; coins it adds are onboarded at once.

**Dashboard refresh.** Every `save_signals` touches `reports/signals_stamp`; the dashboard polls `/api/pulse` (file
mtimes only, no database) every 5 seconds and, when the stamp or `reports/wake_events.json` changes, reloads the list
(notifications) and quietly reloads the open chart. The full refresh (meta, scan, journal) stays once a minute.
The dashboard itself watches the same files and recomputes the list in a background worker process right after new
scores, so the browser's request is answered from the cache (~0.2 s on the server instead of ~7 s).

**Training tickers outside the list.** Every 3 hours the screener also refreshes 15m/5m exchange data (Binance, Bybit,
OKX) and 15m liquidations for `tickers.txt` coins that are not in the list. They are not scored; the point is history
that the sources do not keep for long (OKX open interest ~14 days, Coinalyze 15m liquidations 1–2 months), so that
OKX can later be tested as a model input.

The filter is deliberately looser than the model: its job is not to miss a pump; the model decides whether it is one.
An auto coin stays `ttl_hours` (48) after its last trigger or model signal, at most `max_auto` (30) coins. Alerts,
not list entries, trigger notifications.

**WOKE channel** (since 2026-10-08, `experiments/wake_channel.py`, `reports/wake_channel.md`). The liquidity floor (30-day
average turnover) hid quiet coins that suddenly came alive (OGN on 10-07: +96% after the trigger, never in the list). A
second channel catches exactly them — all at once on the 15m scan:

- 30-day average turnover < `min_turnover` of the active preset (a quiet coin);
- turnover pace = 4-hour volume × 6 ≥ 20 M$/day;
- 4-hour volume ≥ ×10 the median 4-hour volume of the previous 7 days;
- price over 4 hours ≥ +5%, OI (USD) over 4 hours ≥ +20%.

On 29 days of 449 quiet coins: ~10 events a day, 42% gave +10% within 24 h (random bars 12%), 21% gave +20%, but the
median price 24 h later was −5% and 20% of the move was already done at the trigger — a "where to look" tool, not an entry.
The owner's own system longs on such coins: 24 of 26 were caught, a median 9.5 h before his entry.

Mechanics (`scanner.woke_passes` / `woke_hits` / `record_woke`, `apply`): the scan also checks coins whose 24 h turnover is
≥ pace / 6; WOKE events go to `reports/woke_events.json` for every coin, manual ones included (`first`, `last`, reasons and
metrics); auto records get `channel: "woke"` (old records = `"liquid"`) with their own cap `WOKE_CAP` = 15 (the stalest by
last trigger drops out) and TTL 48 h, which a model signal extends (`scanner.touch`). A repeat trigger extends it but is not a
new event. A WOKE coin that passes the regular (liquid) filter moves to the liquid channel. × blocks either channel until a
new trigger. Thresholds are constants (`scanner.WOKE`), not in the presets.

**Implausible OI is skipped (since 2026-10-16).** The 4-hour OI change (both channels) is the sum of log steps between
neighbouring 15m snapshots, without steps where OI in coins changed by more than the bar's traded volume allows
(`features.impossible_oi`) or by more than ×2 / ÷2 in one bar (`OI_STEP_MAX`; real steps that large happen only right
after a listing). With clean data it equals `oi[-1] / oi[-1 − 16] − 1` (checked on 20 coins and all 31 recorded WOKE
triggers). On 2026-10-15, 11:20–18:07 UTC, Binance's OI feed returned wrong values for several hours: WOKE reported EPIC
+223%, GOAT +71%, ARC +202% while the exchange's own history shows +5%, −20% and +11%, so three coins entered WOKE by
mistake.

Presets on the 56 labeled pumps (`python scanner.py --recall`, hourly data from the database, same 4-hour windows; "on time" = the coin
would have been in the list before the impulse started):

| Preset | Turnover ≥ | Volume × / OI % / price % | On time | Only during | Never |
|---|---|---|---|---|---|
| Strict | 30 M$ | 4 / 15 / 3 | 24 | 20 | 12 |
| Balance | 20 M$ | 3 / 10 / 2 | 30 | 17 | 9 |
| Wide | 5 M$ | 2 / 6 / 1 | 47 | 9 | 0 |

The turnover floor costs most: many targets are quiet small coins until the pump starts (10 M$ with the Balance
triggers gives 37/56). At the first live scan the presets passed 3 / 5 / 9 coins. (The recall table was computed with the
24 h turnover; `--recall` now uses the average over its 8-day hourly window.)

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

**Loading indicator.** A thin bar runs along the top of the dashboard and the labeling tool while any request waits for
the server (background polls excluded); a block that is being refreshed (journal, chart, a backtest run) dims and shows
"Оновлюю…". A running backtest shows its steps (coins → data → model → summary) with a progress bar.

**Backtest tab.** Runs of `backtest.py`: launch with a turnover band (from / to), window, model and entry level; KPI cards,
the cumulative result by day, trades and per-coin results. *Порівняти з* puts a second run next to the open one: metrics
side by side with the difference, the second curve dashed, and per-coin differences — fair only on the same coin band.

**Candidate notifications.** Under the notification toggle, a second one also announces Candidates (the 15m model
crossed its threshold, the hourly model has not confirmed yet): one lower tone instead of two, and a toast that says to
check the order book. Candidates come about 15 minutes earlier than Alerts with roughly 1.5× the false alarms.

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

**List width (since 2026-10-19).** In the full dashboard the border between the chart and the ticker list is a handle: drag it to make the list wider or narrower (300 px up to what leaves the chart ~420 px), ← / → when it is focused; the width is remembered per browser (`localStorage` `pd-list-w`), and a double-click restores 392 px. Charts resize themselves. Not shown on the list-only page, in the app's chart window or on narrow screens.

**Section order (since 2026-10-19).** The list sections (NOW, TWAP · HL, WOKE UP, WATCHLIST, QUIET) can be reordered by dragging a section header (the ⋮⋮ grip shows on hover) or with Alt + ↑ / ↓ on a focused header; double-clicking the grip restores the default order. The order is remembered per browser and per PD Watchlist app (`localStorage` `pd-sec-order`), and ↑ / ↓ through tickers follows it.

**TWAP · HL section (since 2026-10-18).** Between NOW and WOKE UP the list shows active Hyperliquid TWAP orders as cards
(`/api/twaps` ← `hl_twap.py`): coin, side and type (`BUY · open`, `SELL · close` = reduce-only), **% Vol** — the TWAP's
dollars per minute divided by the coin's volume per minute over the last hour on Binance (`trade_bars`) and Hyperliquid
together, size and duration, `Slice $X/30s · N% of depth` (one slice vs the Hyperliquid book within ±0.3% on the side it
hits), a progress bar, `Filled`, time left, seconds to the next slice and `Δ Px` since the TWAP started. Filters `Min N% vol`
(default 3) and `Binance only` (default on) are remembered in the browser. A coin of the list with a TWAP above the filter
gets a `T` glyph after the ticker (green buy, red sell); clicking a card opens the coin if it is in the list.

**List-only page (since 2026-10-15).** `#list` shows only the ticker list (with the Sound control and a link to the
full dashboard) and loads no charts; clicking a ticker opens it in the full dashboard (in the PD Watchlist app — in the
app's own chart window; Shift + click — a new additional chart window). It is the page behind the PD Watchlist window (`watch_app/`), and it also works in a narrow
browser window. `#chart/TK` is a light forecast view without the side panel and the list (the app's chart windows): one chart pane and
nothing above it — the ticker, 24 h change, status, P, the model's position and stop and a short Mover note sit in a
translucent legend in the chart's top-left corner, and the only controls (timeframe and volume-profile period) are compact
buttons in the top-right corner; the OI / volume data row is not shown, footprint and the Δ pane are off, the Market / Movers /
Spread requests are not made; ↑ / ↓ still step through the list's tickers; no sounds or notifications.

**Charts tab (since 2026-10-13).** `#charts` shows every ticker of the list's NOW and WOKE UP sections as a grid of
mini charts: the last 96 closed 15m candles, Alert / Candidate zones, the model's entries and the stop of an open trade
(main mode), with the same state chip and glyphs as the list row, P and the 24 h change. The data comes from
`/api/minicharts?symbols=…`, which reads the list cache (`dashboard.mini_charts` builds it together with the list from
bars that are already loaded, so the tab adds almost no database work; `/api/list` does not carry it). The last candle
is live through one combined Binance WebSocket for the whole grid; charts are plain SVG. A switch selects the
timeframe: 15m (24 h) and 1h (2 days, aggregated from the cached 15m bars; a bar shows the highest signal level of its
hour) come from the list cache, 5m (8 h) takes one extra candle query for all tickers of the grid (cached 20 s; a 5m bar
shows the level of its 15m bar). Sections collapse (a collapsed
section is not loaded; WOKE UP is collapsed by default). Keys: PageDown from Forecast opens the tab, PageUp returns
(Up / Down on Forecast already step through the list), arrows move the selection, Enter or a click opens the ticker in
Forecast.

**One or two chart windows (since 2026-10-06).** The toolbar above the chart has ▢ / ▢▢ (one window or two side by
side), **Full info** and **Period ▾** (1d / 3d / 7d / 30d). The *active* window (orange frame; click it or press Tab)
takes the ticker clicked in the list and ↑/↓, and is the one the left panel describes; the list marks the other
window's ticker faintly. Each window's header shows ticker, price, 24 h change, status, P and — with Full info — the
open position and its stop. **Full info off** asks the server for a light response (`/api/ticker?light=1`: no position
simulation, order book, wake-up history, scanner marks or OI share), about 2–3× faster; the left panel then says that
the position needs Full info. The left panel collapses to a 56 px rail (‹ button or `[`): section icons plus the
active window's ticker, P, status dot and position; the state is remembered per layout (collapsed by default with two
windows). `/api/ticker` responses are cached until the next screener scores (and at most 30 s), so switching back and
forth between tickers is instant.

**Forecast** — three columns:

- *left*: model switch; pump probability with a threshold bar and the signal's age in hours; OI change over
  4 h, volume z-score, share of long accounts; the model's simulated position with stop and exit
  state;
- *centre*: chart ([TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts),
  Apache 2.0, local copy in `app/vendor/`): candles with signal zones, entry and exit markers of every
  simulated trade, the stop line, OI candles (in coins by default, USD on a toggle), volume and the
  hourly probability with the threshold;
- *right*: the watchlist like a trading terminal — search, ★ favourites only, **Refresh**, sections NOW / WOKE UP /
  WATCHLIST / QUIET (see Watchlist), sorting by probability, OI change over 24 h or price change over 24 h within each
  section, a 24-hour probability sparkline per ticker, and the latest signal episodes.

**Ruler** (like TradingView's measure tool; shared `app/js/ruler.js`, also in the labeling tool): Shift + click on the
chart or the ruler button, move, click to fix, click again or Esc to remove. Shows the price change, bars and
time, summed volume, OI change in USD and in coins, and the model's max probability with hours above threshold.

**Volume profile** (like ATAS; shared `app/js/profile.js` + `volume_profile.py`, also in the labeling tool):
**День / Тиж** draws a profile for every UTC day or week (Monday start) at the left of its candles; the profile
button or **V** draws a fixed-range profile — click the start, click the end; edges can be dragged, × or Del (cursor
over it) removes it; fragments are saved per ticker in the browser and survive timeframe switches. Each profile shows
volume per price level (value area brighter), the POC (orange; a fragment's POC extends to the right as a dashed line
to watch retests) and VAH / VAL (dashed). Built from 5m candles (15m / 1h when 5m don't cover the range), so it
approximates a tick profile within a level or two.

Keyboard: **↑/↓** next ticker, **Enter** refresh the ticker, **F** favourite. **Alert notifications**: when a new
live signal episode starts while the page is open, a sound, a browser notification and an in-page toast
(opt-in; favourites, sort order and settings are stored in the browser). All sounds live in one **Sound ▾** button next to
Movers in the top bar (since 2026-10-08; the three buttons in the right panel were removed to give the list more room): five
independent switches — Alert (model entry: two high tones + browser notification), Re-alert (Alert again inside an open model
trade: one short high tone, off by default), Candidate (one low tone, only while the model has no open trade), Volume 5m
(voice) and Woke (the first WOKE trigger: two rising tones + a notification with the reason). The button shows how many are
on ("2/5", "off"). `/api/list` splits the live Alert episodes with `dashboard.split_by_trades`: `alerts` = model entries
(`model_trades` runs the same `risk.position_state` as the chart; a re-entry after a stop inside a continuous Alert is added
with `reentry: true`), `realerts` = episodes inside an open trade (with `trade`: entry time and price, current result).

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
- scope: live + history (default: every out-of-sample bar — this is how the model itself is judged; history = bars scored later, e.g. the screener was off or the scanner added the coin later, same decision on the same data), live only (what reached you in time), + in-sample; the choice is remembered; period 7 / 30 days / all, or "Dates…" — a calendar picker (two months, range in two clicks, presets today / yesterday / this or last week / this or last month / this year; UTC days by trade entry, `/api/journal?from=&to=`);
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

**Stop shown without waiting for the bar close (2026-10-12).** `risk.position_state` works on closed 15m bars, so a stop
used to appear only after the bar closed and the screener cycle ran (up to ~15 minutes later). Two layers now bring it
forward; the trade result is unchanged because the exit price is the stop level in every case.

| Where | How | Delay |
|---|---|---|
| Open chart | the browser compares the live candle (exchange WebSocket) with the open position's stop: header "стоп спрацював −x%", the stop line turns red, the list row shows `Stop −x%` (`P.stopLive`, `st.liveStops`) | seconds |
| List, header and chart for every ticker | `dashboard.live_stop` → `risk.intrabar_stop`: 5m bars closed after the last 15m bar; a low at or below the stop closes the trade (`intrabar: true`, exit bar = the 15m bar it happened in) | up to ~5 min |

The half-exit target is not checked intrabar: inside one bar `position_state` checks the stop before the target, and
5m bars could give a different order than the closed bar will. Journal and backtest are untouched.

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
- **Risk on 15m:** stop = entry − k·ATR(14 bars of 15m), k = 90th percentile of the drawdown from the model's actual
  entry (~2.5); no trailing stop (removed 2026-10-01: the owner exits by the order book). **Partial exit** (since
  2026-10-02, `risk.PARTIAL_EXIT`): half of the position is closed at entry + 3·ATR, then the stop on the rest moves to
  breakeven; the rest exits when the Alert is gone for 4 bars (1 hour) or at the stop. A trade's result is that of the
  whole position. On the backtest coins outside training this raised the profit factor from 0.74 to 0.86, and on the
  screener journal from 1.44 to 1.67 (`experiments/exit_partial.py`). The chart shows the partial exit ("½ +x%"),
  the target line before it and the breakeven stop after it. Hourly models keep the plain exit.
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
