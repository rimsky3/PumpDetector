# Trade flow: exchange archives and own feed

## Status (2026-10-07)

- **History comes from the exchanges' public archives** — every trade, back to 2020, published the next day:

  | Exchange | Archive | Columns |
  |---|---|---|
  | Binance | `data.binance.vision/data/futures/um/daily/aggTrades/<SYMBOL>/<SYMBOL>-aggTrades-<YYYY-MM-DD>.zip` (also `monthly/`) | `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time` (ms), `is_buyer_maker` — the same aggTrades as the live stream |
  | Bybit | `public.bybit.com/trading/<SYMBOL>/<SYMBOL><YYYY-MM-DD>.csv.gz` | `timestamp` (s), `symbol, side` (taker), `size, price, tickDirection, trdMatchID, grossValue, homeNotional, foreignNotional, RPI` |

  So flow and footprint features can be built for any past period, including every labeled impulse — no need to wait
  for a recording to accumulate. OKX has no archive at a plain URL (and the model does not use OKX).
- **The live feed (`trade_recorder.py`) is ready and tested but not running.** It is needed only for live flow features
  (the archive is a day late; REST cannot replace it: Bybit returns only the last 1000 trades, Binance pages of 1000 at
  a high request weight). It sits behind the compose profile `trades`; start it once an experiment on the archives shows
  the features help: `docker compose -f docker-compose.server.yml --profile trades up -d traderec`. The 7 days of history
  that z-score features need at start come from the archive.

## Own feed

`trade_recorder.py` (container `traderec`) records every trade on Binance and Bybit USDT perpetuals and stores 5-minute
summaries.

Why: candles only give volume, and the trade count in them is the same information
(`experiments/trade_count.py`: trade-count features replaced the volume features instead of adding to them, and the
backtest did not improve). What candles do not show is the *composition* of the flow: how much came from large trades,
which side was the aggressor, the largest single trade. Exchanges keep only a few days of trade history, so this has
to be recorded going forward. One observation that motivated it: at the start of a labeled impulse the average trade
is clearly larger than usual (z ≈ +1.2…1.4), i.e. larger players arrive rather than a crowd of small orders.

## Streams

| Exchange | Stream | What one trade is |
|---|---|---|
| Binance | `wss://fstream.binance.com/market/ws`, `<symbol>@aggTrade` (the old `/ws/` path is silent) | fills of one taker order at one price merged together — close to "the order size"; `m = true` means the buyer was the maker, so the aggressor sold |
| Bybit | `wss://stream.bybit.com/v5/public/linear`, `publicTrade.<SYMBOL>` | every fill separately; `S` is the taker side |

Because of that difference the two exchanges' trade counts are not directly comparable: compare each exchange with its
own history. All USDT perpetuals are subscribed (≈ 525 on Binance in 2 connections, ≈ 780 on Bybit in 4); the symbol
lists are re-read every 6 hours for new listings. Measured on 2026-10-06 at a quiet hour: ≈ 510 trades/s on Binance and
≈ 370/s on Bybit; the process uses a few percent of one core.

## Table `trade_bars`

One row per exchange, symbol and 5-minute bar (by the exchange's trade time):

| Column | Meaning |
|---|---|
| `n_buy`, `n_sell` | trades where the aggressor (taker) bought / sold |
| `v_buy`, `v_sell` | their value in USD |
| `max_buy`, `max_sell` | the largest single trade in the bar, USD |
| `buy_n[]`, `sell_n[]`, `buy_usd[]`, `sell_usd[]` | counts and USD by trade-size bucket: `[1]` < $500, `[2]` $500–2.5k, `[3]` $2.5k–10k, `[4]` $10k–50k, `[5]` $50k–250k, `[6]` ≥ $250k (PostgreSQL arrays start at 1) |

A bar is written once, after it closes (+5 s for late messages), and only if the connection serving that symbol was
alive for the whole bar — a bar where the connection (re)started midway is dropped rather than stored incomplete.
`trade_ws_coverage` keeps when each connection was alive (`started … last_seen`, heartbeat every minute): a missing row
means *no data* or *no trades*, and coverage tells which.

Raw trades are not stored: the whole market would be tens of GB per month. The 5-minute summaries are ≈ 2–3 GB per
month (the server has ≈ 54 GB free as of 2026-10-07).

Local check (2026-10-06, one full bar): USD volume and the taker-buy share of every bar matched the exchanges' 5-minute
candles to the dollar (Binance and Bybit).

## Usage and next steps

- `python trade_recorder.py --stats [--hours N]` — bars, symbols, trades, value, trades ≥ $250k per exchange, and
  connection coverage.
- Next: an experiment on the archives with flow features — share of large trades in volume against the coin's own
  history, aggressor imbalance by size, largest trade against the norm, and footprint features (delta at the bar's high,
  absorption, where in the bar the volume sat) — in the 15m model, by leave-ticker-out and the backtest. If they help:
  start the live feed (adding the footprint to it) and build the history from the archives.
