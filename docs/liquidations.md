# Own liquidation feed (WebSocket)

Since 2026-10-06 the server records liquidations straight from the exchanges (`liq_recorder.py`, container `liqrec`).
Goal: drop the slowest part of the screener cycle — Coinalyze 15m liquidations (~50 calls for the hot coins at 40 calls
per minute) — so that a hot coin is scored ~30–60 s after its 15m bar closes instead of ~3 minutes.

## Streams

| Exchange | Stream | Notes |
|---|---|---|
| Binance | `wss://fstream.binance.com/market/ws/!forceOrder@arr` (the old `/ws/` path is silent) | whole market in one stream, but **at most one liquidation per symbol per second** — cascades are undercounted |
| Bybit | `wss://stream.bybit.com/v5/public/linear`, `allLiquidation.<SYMBOL>` | one topic per symbol (~780 USDT perpetuals, 4 connections of 200); all liquidations, pushed every 0.5 s |
| OKX | `wss://ws.okx.com:8443/ws/v5/public`, `liquidation-orders` / `SWAP` | whole market; size in contracts → coins via `ctVal` (`public/instruments`); also capped at one per contract per second |

Hyperliquid (present in Coinalyze) and other venues are not recorded.

Inverse (coin-margined) contracts — OKX `*-USD-SWAP` (`ctType` inverse, `ctVal` in USD) and Binance COIN-M (`BTCUSD_PERP`
and others, which the Binance stream also delivers) — are sized in USD per contract (BTC $100, others $10), not in coins:
`usd = contracts × contract size`, `qty = usd / price` (sizes from OKX `public/instruments` and Binance
`dapi/v1/exchangeInfo`). Until the 2026-10-07 fix they were multiplied by the price once more (one OKX BTC-USD-SWAP event
read $14 B instead of $17 M); rows recorded before the fix for these symbols are inflated. USDT contracts — the only ones
the model uses — were always correct.

## Tables

- `liq_events` — one row per liquidation: `exchange`, `symbol` (the exchange's own symbol, = `markets.symbol_on_exchange`),
  `ts` (exchange time, ms), `side` (`long` = a long position was force-closed, i.e. a forced sell; `short` the opposite),
  `qty` (coins), `price`, `usd`, `recv_at` (when the message arrived; stream latency = `recv_at − ts`, ~0.3–1.5 s).
- `liq_ws_coverage` — when each connection was alive (`started … last_seen`, heartbeat every minute). A gap between
  sessions means *no data*, not *zero liquidations* — the same rule as `liquidation_coverage` for Coinalyze.

The whole market is recorded, not only the watchlist, so a coin that joins the list later already has history.

## Status and next steps

1. **Recording** (from 2026-10-06 15:14 UTC on the server). `python liq_recorder.py --stats [--hours N]` shows events,
   sums, stream latency and connection coverage.
2. **Comparison with Coinalyze** after 1–2 weeks: sum our events into the same 15m bars per exchange and compare with
   Coinalyze 15m liquidations for the watchlist (correlation, ratio, the model's liquidation features and P at Alerts).
3. **Switch** if they match (or the difference is stable and can be corrected): the 15m score takes liquidations from
   `liq_events` instead of Coinalyze. The model's liquidation features use only 1 h and 4 h windows, so a few hours of
   recording are enough after a restart — no week-long warm-up.
