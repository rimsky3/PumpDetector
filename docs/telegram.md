# Telegram bot

`tg_bot.py` (server container `tgbot`, since 2026-10-06) sends signal notifications to the phone and answers a few
buttons. It is one more client of the dashboard, like the browser: signals come from `/api/list` (the same cached list),
ticker details from `/api/ticker`, results from `/api/journal`. It computes nothing itself, so its numbers are the
dashboard's numbers. The screener and the dashboard are unchanged.

## Notifications

| Type | Default | Message |
|---|---|---|
| 🔴 Alert | on | **model entry** only (since 2026-10-08): the model has no open trade and opens one — the same trade as the "Вхід" mark on the chart (`risk.position_state`, the main mode — `"main": true` in `strategies.json`, read on every message, `split` since 2026-10-12), including a re-entry after a stop. Ticker, price, 24 h change, P (15m) and the hourly confirmation, OI change over 4 h, entry, stop (price and %), the half-exit target, the limit hint, **Drivers** (top 3 features behind P, the strongest one against it, and `⚠ No volume behind this signal` when volume pulls P down), the 15m bar |
| ⚪ Late | on | an Alert episode **without a model entry** in the main mode (2026-10-12): in `split` the start model and the hourly confirmation did not meet on one bar (CHIP 10-10), or a repeat signal after a stop. The model does not take it (on average weaker: profit factor 1.31 vs 1.48); the message gives the same levels as Alert with "stop if you enter now" and leaves the decision to the owner. Source: `/api/list → lates` |
| 🔁 Re-alert | off | Alert fired again while the model trade is still open (before 2026-10-08 these came as ordinary Alerts): P, the trade's entry time and price, its current result, the bar |
| 🟠 Candidate | off | the same as Alert without a position: the stop "if entering now" (price − k·ATR). Only when the model has no open trade and there was no Candidate / Alert in the previous hour |
| 🟣 Volume 5m | on | volume × weekly norm, 15-minute price change, hourly OI change |
| 🟪 WOKE | on | the first trigger of the scanner's WOKE channel (docs/screener.md): a dormant coin came alive — average turnover → current pace, 4-hour volume × norm, price and OI over 4 hours, the model's P against its threshold, and the note that it is not an entry signal (42% gave +10% within a day, the median day was −5%). In Ukrainian, like the rest of the bot |
| 🔷 Spread | on | prices of one ticker diverged between exchanges (`spread.py`, since 2026-10-13; docs/screener.md): which exchange is more expensive and by how much (bid against ask), how long it has held, the coin's normal spread, mid prices and funding per exchange. Source: `/api/spreads`, polled every 5 s; thresholds and the coin scope are set in the dashboard ("Spread ▾"). More than 5 new events in one pass are sent as a single summary (an exchange glitch, not 5 opportunities). For a coin outside the dashboard list the Chart button is replaced by "add to watchlist" |
| ⏳ TWAP | off | a new Hyperliquid TWAP order passed the filters of the dashboard's TWAP · HL section (since 2026-10-19): side and type (open / close), % of the coin's volume, size and duration, start time, slice vs order book depth, price change since the start. Source: `/api/twaps` → `events`, polled every 5 s; the filters (`Min % vol`, `$ K`, `Binance only`, `🔔 open only`) are the same as in the dashboard (`twap_config.json` on the server). Each TWAP is sent once; TWAPs already active when the dashboard or the bot starts, or when the filters are changed, are not sent. More than 5 new TWAPs in one pass — one summary. Coins without a Binance perpetual get only the mute button |
| 🟢 b / p | on | footprint pattern (`footprint.py`, docs/trades.md): **b** — a stop-run below the 2-hour low that was bought back, with the pierced low, the share of the bar's volume below it and its delta, volume × norm, the model context (Alert / Candidate within 6 h) and the stop under the bar's low; **p** — the same at the highs (a warning for longs) |

One message per signal episode (the same episodes as the dashboard's sound notifications). An episode continues while the
next signal row (a regular 15m bar or a rolling 5-minute window) comes within one bar of the previous signal row; before
2026-10-08 any non-signal rolling row in between split it, so one continuous Alert could notify every 15 minutes. On start the bot only
remembers the current episodes, and episodes older than 45 minutes after their bar closed are never sent, so a restart
or downtime does not flood the chat. A Candidate is skipped when the same ticker has an Alert in the same batch.

Buttons under a signal: **📈 Chart** (opens the ticker in the dashboard), **🔕 mute the ticker for 2 h**, and for coins the
scanner added, **📌 add to watchlist** (`/api/watchlist/add`).

## Buttons and commands

The chat keyboard has **📋 Status** (current Alerts and Candidates), **📊 Results** (today / yesterday / week / month:
closed trades, win rate, profit factor, sum, average, stops, best and worst trade, open trades — the journal's
"live + history" scope for the main mode; plus the b / p marks of the period from `/api/fp_stats`: how many, how many
finished, reached +2R / stopped / 12 hours, average R) and **⚙️ Settings** (toggle each notification type, list muted tickers).
`/status`, `/results` and `/settings` do the same.

## Privacy and state

The first chat that sends `/start` becomes the owner (or set `TELEGRAM_CHAT_ID`); the bot answers nobody else. State —
owner, settings, muted tickers, already-sent episodes, the update offset — lives in `reports/tg_state.json`.

## Setup

1. @BotFather → `/newbot` → token into `.env` as `TELEGRAM_BOT_TOKEN=…` (laptop), then onto the server's `.env`.
2. `docker compose -f docker-compose.server.yml up -d tgbot`; `python tg_bot.py --check` checks the token and the
   dashboard connection.
3. Open the bot in Telegram and press **Start**.
