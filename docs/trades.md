# Trade flow: exchange archives, own feed and footprint marks

## Status (2026-10-07)

- **History comes from the exchanges' public archives** — every trade, back to 2020, published the next day:

  | Exchange | Archive | Columns |
  |---|---|---|
  | Binance | `data.binance.vision/data/futures/um/daily/aggTrades/<SYMBOL>/<SYMBOL>-aggTrades-<YYYY-MM-DD>.zip` (also `monthly/`) | `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time` (ms), `is_buyer_maker` — the same aggTrades as the live stream |
  | Bybit | `public.bybit.com/trading/<SYMBOL>/<SYMBOL><YYYY-MM-DD>.csv.gz` | `timestamp` (s), `symbol, side` (taker), `size, price, tickDirection, trdMatchID, grossValue, homeNotional, foreignNotional, RPI` |

  So flow and footprint features can be built for any past period, including every labeled impulse. OKX has no archive
  at a plain URL (and the model does not use OKX).
- **Experiments on the archive** (notes/experiments.md, 2026-10-07): trade-composition and footprint features in the
  15m model — within noise (`experiments/flow_archive.py`); the owner's "stop-run that gets bought back" pattern on all
  bars — no edge (`experiments/spring_pattern.py`); **the same pattern only when the model already sees the coin in play
  (a Candidate or Alert within 6 hours) — +2R before the stop in 38% of cases against 29% for the model's own entry,
  +0.07R per trade, positive in both halves of the month** (`experiments/spring_context.py`, 218 coins outside
  training). A small edge, one month, within about one standard error of zero — so it ships as a hint on the chart
  with its own live statistics, not as a model change.
- **The live feed (`trade_recorder.py`) is needed for the b / p marks** — the archive is a day late, and REST cannot
  replace a live feed (Bybit returns only the last 1000 trades, Binance pages of 1000 at a high request weight). It runs
  as the server container `traderec` (since 2026-10-07; `deploy.sh --restart` starts it).

## Own feed

`trade_recorder.py` (container `traderec`) records every trade on Binance and Bybit USDT perpetuals and stores 5-minute
summaries with a footprint.

Why: candles only give volume, and the trade count in them is the same information
(`experiments/trade_count.py`: trade-count features replaced the volume features instead of adding to them). What
candles do not show is the *composition* of the flow — large vs small trades, the aggressor side, the largest trade —
and *where inside the bar* it traded (the footprint). At the start of a labeled impulse the average trade is clearly
larger than usual (z ≈ +1.2…1.4): larger players arrive rather than a crowd of small orders.

## Streams

| Exchange | Stream | What one trade is |
|---|---|---|
| Binance | `wss://fstream.binance.com/market/ws`, `<symbol>@aggTrade` (the old `/ws/` path is silent) | fills of one taker order at one price merged together — close to "the order size"; `m = true` means the buyer was the maker, so the aggressor sold |
| Bybit | `wss://stream.bybit.com/v5/public/linear`, `publicTrade.<SYMBOL>` | every fill separately; `S` is the taker side |

Because of that difference the two exchanges' trade counts are not directly comparable: compare each exchange with its
own history. All USDT perpetuals are subscribed (≈ 525 on Binance in 2 connections, ≈ 780 on Bybit in 4); the symbol
lists are re-read every 6 hours for new listings. Measured at quiet hours: ≈ 330–510 trades/s on Binance and ≈ 300–370/s
on Bybit; the process uses a few percent of one core.

## Table `trade_bars`

One row per exchange, symbol and 5-minute bar (by the exchange's trade time):

| Column | Meaning |
|---|---|
| `n_buy`, `n_sell` | trades where the aggressor (taker) bought / sold |
| `v_buy`, `v_sell` | their value in USD |
| `max_buy`, `max_sell` | the largest single trade in the bar, USD |
| `buy_n[]`, `sell_n[]`, `buy_usd[]`, `sell_usd[]` | counts and USD by trade-size bucket: `[1]` < $500, `[2]` $500–2.5k, `[3]` $2.5k–10k, `[4]` $10k–50k, `[5]` $50k–250k, `[6]` ≥ $250k (PostgreSQL arrays start at 1) |
| `fp_lvl0`, `fp_buy[]`, `fp_sell[]` | footprint: aggressive buy / sell USD at price levels 0.1% apart — level = floor(ln price / ln 1.001) (`footprint.level`), `fp_buy[k]` is level `fp_lvl0 + k − 1`. One grid for every bar and coin, so a 15m bar is the sum of its three 5-minute bars |

A bar is written once, after it closes (+5 s for late messages), and only if the connection serving that symbol was
alive for the whole bar — a bar where the connection (re)started midway is dropped rather than stored incomplete.
`trade_ws_coverage` keeps when each connection was alive (`started … last_seen`, heartbeat every minute): a missing row
means *no data* or *no trades*, and coverage tells which.

Raw trades are not stored. With the footprint the summaries are ≈ 4–6 GB per month for the whole market (the server
has ≈ 54 GB free as of 2026-10-07); anything older can be rebuilt from the archive.

Checks: USD volume and the taker-buy share of every bar matched the exchanges' 5-minute candles to the dollar (Binance
and Bybit, 2026-10-06); the footprint of every bar sums to the bar's volume (BTC, ORCA, 2026-10-07).

## Footprint marks b / p (`footprint.py`)

One module for the experiments, the screener, the dashboard and the bot — the rule and the outcome live only there.

- **b** ("залочка"): the 15m bar pierces the low of the previous 2 hours; ≥ 30% of the bar's volume traded below that
  low (stops, liquidations, shorts on the breakdown); the bar closes back above it; ≤ 20% of the volume sits in the top
  third of the bar (volume at the bottom — the "b" shape); volume ≥ 2× the previous day's median. Read: a large buyer
  held the low with limit orders. Examples confirmed with trades: ORCA 2026-10-06 20:00, API3 2026-10-05 14:30, CAP
  2026-10-06 03:15.
- **p**: the mirror at the highs — buyers above the 2-hour high were absorbed and the bar closed back below it.
- **Only in context**: the screener checks a coin only if it had a Candidate or Alert within the last 6 hours. Outside
  that context the same pattern is noise (≈ a random bar), and stricter thresholds (volume ×8, ≥ 60% below the low) are
  worse, not better — a huge volume below the low is more often real selling.
- **Outcome of a mark** (`footprint.outcome`, the same function as in the experiments): entry at the bar's close, stop
  0.1% beyond the bar's low (b) / high (p), exit at +2R, at the stop, or after 12 hours; fees 0.1%.
- **Where it shows**: `fp_marks` table (written by the screener after the 15m scores), the dashboard chart (green "b"
  under the candle, orange "p" above it; hover for the pierced level, the share of volume beyond it and its delta,
  volume × norm, the context, the stop and the outcome), Telegram (setting "b / p"), and the bot's Results
  (`/api/fp_stats`). The live statistics of the marks are the forward test of the backtest result above.

## History from the archive (`trade_backfill.py`)

The recorder only sees what happens after it starts. `trade_backfill.py` replays the daily Binance archive into the
same `trade_bars` rows (identical to the recorder's aggregation — checked field by field on API3 2026-10-05 14:30), so
the footprint chart and the b / p marks work on history exactly as live. Historical marks are stored with
`live = false`: dim on the chart, never sent to Telegram, and counted on a separate line in the bot's Results — live
marks are the real test on new data, historical ones repeat the backtest. The screener runs it as a separate
low-priority process (at start for all list coins over 30 days, for coins newly added to the list, and daily after
03:00 UTC for yesterday — which also fills gaps when the recorder was down). Only Binance; ≈ 40 coins × 30 days is a few
hundred MB. Trial on 5 coins × 10 days: 9 historical marks (≈ one per coin every 5 days).

## Footprint on the chart (FP button)

The dashboard's FP button (next to the ruler) draws the footprint inside each candle: aggressive sells to the left of
the centre (red), buys to the right (green), per 0.1% price level; the bar's POC is outlined; numbers appear when candles
are wide enough. Candles become hollow while FP is on (as in ATAS clusters). Data come from `/api/footprint` (5-minute
`trade_bars` summed to the chart's timeframe) only for the visible bars and only while FP is on; when the chart is zoomed
out the levels are not drawn (a hint says to zoom in). Live recorder data for every coin; history for list coins.

## Entry model — tried, rejected (2026-10-07)

`experiments/entry_model.py`: a LightGBM on the context bars only (84,749 bars, 215 coins), target "+2R before a stop under
the bar's low (≥ 0.5 ATR) within 12 h", 31 footprint / flow / model-state features, trained on half the coins in the
first half of the period and tested on the other coins in the second half (both ways). AUC 0.499 / 0.505 — nothing
learned; its top 10–33% of bars gave −0.07…−0.08R, like a random context bar (−0.15R). On the same test, **rule b gave
+0.08R, +2R in 38%, PF 1.13, positive in both folds** (+0.15 / +0.02R), and the main model's entry −0.18R. The simple rule
from the owner's practice holds out of sample; a generic model on the same data does not.

## Usage and next steps

- `python trade_recorder.py --stats [--hours N]` — bars, symbols, trades, value, trades ≥ $250k per exchange, and
  connection coverage.
- `python trade_backfill.py [BASE …] [--days N | --yesterday]` — footprint history and historical marks from the archive.
- Next: the 6-month check of the b pattern on the monthly Binance archives (notes/experiments.md, ideas), and the live
  statistics of the marks after 2–3 weeks.
