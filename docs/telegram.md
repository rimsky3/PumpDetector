# Telegram bot

`tg_bot.py` (server container `tgbot`, since 2026-10-06) sends signal notifications to the phone and answers a few
buttons. It is one more client of the dashboard, like the browser: signals come from `/api/list` (the same cached list),
ticker details from `/api/ticker`, results from `/api/journal`. It computes nothing itself, so its numbers are the
dashboard's numbers. The screener and the dashboard are unchanged.

## Notifications

| Type | Default | Message |
|---|---|---|
| 🔴 Alert | on | ticker, price, 24 h change, P (15m) and the hourly confirmation, OI change over 4 h, entry, stop (price and %), the half-exit target, the limit hint, the 15m bar |
| 🟠 Candidate | off | the same without a position: the stop "if entering now" (price − k·ATR) |
| 🟣 Volume 5m | on | volume × weekly norm, 15-minute price change, hourly OI change |
| 🟢 b / p | on | footprint pattern (`footprint.py`, docs/trades.md): **b** — a stop-run below the 2-hour low that was bought back, with the pierced low, the share of the bar's volume below it and its delta, volume × norm, the model context (Alert / Candidate within 6 h) and the stop under the bar's low; **p** — the same at the highs (a warning for longs) |

One message per signal episode (the same episodes as the dashboard's sound notifications). On start the bot only
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
