# Commands

Every script in the project with its options and when to use them. `python <script> --help` prints the same
options straight from the code. Run everything from the project folder with the virtual environment active.

## Typical workflows

| Task | Commands |
|---|---|
| Add a ticker for training | in `python label.py`: find it under **Усі Binance** → **Завантажити** → label it → after reviewing the whole ticker press **У навчання** → `python features.py` → `python train.py` → restart `screener.py` and `dashboard.py` |
| Confirm a journal signal | open the ticker in `label.py`, label the interval as `pump` or `false_start` → `python features.py` (only that interval enters training) |
| Scan a coin live only | dashboard: type it in search → **＋ Додати … у watchlist** (or the field in the **Сканер** card); data and scores load at once. Or add a line to `watchlist.txt` — the screener re-reads it every cycle |
| Drop a coin from scope | `label.py`: open it → **Виключити** (with a reason) — written to `excluded.txt`; it leaves training, the scanner, the screener and the dashboard list |
| Let the market fill the list | the **Сканер** card in the dashboard: pick Суворо / Баланс / Широко / Свій → **Зберегти**; `python scanner.py --recall` refreshes the "on time on labeled pumps" numbers |
| Fix a label | edit it in `python label.py` (or `pumps.csv` + `python load_pumps.py`) → `python features.py` → `python train.py` |
| Weekly data refresh | `python fetch.py` (all tickers; extends history, which the provider keeps for ~2 months) |
| Daily operation | since 2026-10-06 the screener and dashboard run on the server (docs/server.md): double-click `tools/server_dashboard.cmd`. Do **not** start `screener.py` on the laptop as well (shared Coinalyze limit) |
| Ship code / a new model to the server | `bash tools/deploy.sh --restart` / `bash tools/deploy.sh --models --restart` |

Before retraining, label fresh signals on training tickers if they were pumps (see `--fresh-hours`).

## Server

### `tools/deploy.sh` — laptop → server (Git Bash)

| Option | What it does |
|---|---|
| (none) | copy the code: files tracked by git + new non-ignored ones; skips `watchlist.txt` and `scan_config.json` (the server's dashboard edits them) and never copies `.env` |
| `--models` | also `models/` |
| `--restart` | restart the `screener` and `dashboard` containers afterwards and start any compose service that is not running yet (`docker compose up -d`) |
| `--labels` | only the labels and lists (`pumps.csv`, `tickers.txt`, `excluded.txt`, `reviews.csv`), then on the server `load_pumps.py` and `fetch.py --new` (hourly history for new training tickers) |
| `--backtests` | also `reports/backtests/*.json` (shown in the dashboard's Backtest tab) |
| `--yes` | do not ask about uncommitted changes (by default the script lists them and asks; without a terminal it stops) |
| `--first` | initial deploy: also `models/`, the lists, scanner / wake JSON state and backtest runs |

### `liq_recorder.py` — own liquidation feed (server container `liqrec`)

| Option | What it does |
|---|---|
| (none) | record Binance / Bybit / OKX liquidations over WebSocket into `liq_events` (runs forever, reconnects) |
| `--stats` | show what was recorded: events, USD, symbols, stream latency per exchange, connection coverage |
| `--hours N` | window for `--stats` (default 6) |

On the server: `docker compose -f docker-compose.server.yml exec -T liqrec python liq_recorder.py --stats`. See docs/liquidations.md.

### `trade_backfill.py` — footprint history from the Binance trade archive

Replays the daily Binance `aggTrades` archive (the same trades as the live stream) into `trade_bars` in the recorder's
exact format (checked bar by bar against the recorder's own aggregation), then finds historical b / p marks there
(`live = false`). Days that already have ≥ 280 five-minute bars are skipped; recorder rows are never overwritten. The
screener runs it as a separate low-priority process: at start (all list coins, 30 days), for coins newly added to the
list, and once a day after 03:00 UTC for yesterday.

| Option | What it does |
|---|---|
| `BASE …` | coins (default: the screener's list — watchlist + auto) |
| `--days N` | last N full days (default 30) |
| `--yesterday` | only yesterday (the daily run) |
| `--no-marks` | footprint only, no historical marks |

### `trade_recorder.py` — own trade feed (server container `traderec`)

| Option | What it does |
|---|---|
| (none) | record every Binance (`aggTrade`) and Bybit (`publicTrade`) USDT-perpetual trade over WebSocket into 5-minute summaries `trade_bars`: aggressor side, USD, largest trade, size buckets (runs forever, reconnects) |
| `--stats` | show what was recorded: bars, symbols, trades, USD, trades ≥ $250k per exchange, connection coverage |
| `--hours N` | window for `--stats` (default 6) |

On the server: `docker compose -f docker-compose.server.yml exec -T traderec python trade_recorder.py --stats`. See docs/trades.md.

### `tools/pull_db.sh` — server database → laptop (Git Bash)

Dumps the server database, restores it into the **local** database (replacing its contents) and reloads `pumps.csv`
into the `pumps` table. Run before labeling, training or backtests (~4–5 min); Docker Desktop must be running and
`label.py` closed.

### `tg_bot.py` — Telegram bot (server container `tgbot`)

| Option | What it does |
|---|---|
| (none) | run the bot: signal notifications (Alert / Candidate / Volume 5m) and buttons (status, journal results, settings) |
| `--check` | check the token and the dashboard connection without sending anything |

Needs `TELEGRAM_BOT_TOKEN` in `.env`; `DASH_URL` and `PUBLIC_URL` come from docker-compose. See docs/telegram.md.

### `tools/server_dashboard.cmd`

Double-click: SSH tunnel (laptop port 8766 → server port 8766) and the browser on `http://localhost:8766/`. Closing the window closes the tunnel.

### `dashboard.py --host`

Address to listen on (default `127.0.0.1`). In the server container it is `0.0.0.0`; Compose still publishes the port on the server's `127.0.0.1` only.

## Data

### `fetch.py` — Coinalyze → PostgreSQL

| Option | Default | Meaning |
|---|---|---|
| `tickers …` | all of `tickers.txt` | base assets to collect, e.g. `AKE QNT` |
| `--days N` | 60 | days of hourly history to request |
| `--new` | off | only tickers from `tickers.txt` that are not in the database yet, or whose hourly history is shorter than `--days` (minus 3 days) — e.g. a watchlist coin (10 days of hourly data) just moved to training; a coin younger than `--days` is re-fetched on every run |

Re-running is safe: rows are upserted. The API limit counts symbols, not days, so `--days` does not make a run
cheaper; `--new` does.

### `binance_archive.py` — Binance history from the public archive

| Usage | Meaning |
|---|---|
| `python binance_archive.py NMR ZEC --from 2026-07-25 --to 2026-10-05` | download daily archive files from `data.binance.vision` (15m klines with taker buy volume, `metrics`: OI and long/short account ratio every 5 min) and write them into the same raw tables as `fetch_exchange.py` (through `exchanges.normalize`); funding from the API |
| `… --tf 15m 5m` | timeframes (default 15m) |
| `… --check` | compare the archive with what is already in the database (API rows) on common bars; writes nothing |

The API gives Binance OI and long/short for 30 days only; the archive has them since at least December 2021. Verified
against the API: prices and taker buy volume identical, OI identical, long/short equal up to the API's 4-digit
rounding. Two archive quirks are handled: `create_time` is 5 minutes earlier than the API timestamp of the same
snapshot (shifted +5 min), and OI = 0 appears on rare snapshots (dropped). Daily files are cached in
`data/binance_archive/`.

### `fetch_exchange.py` — 15m and 5m from Binance and Bybit

| Option | Default | Meaning |
|---|---|---|
| `tickers …` | all of `tickers.txt` | base assets to collect |
| `--tf 15m 5m` | both | timeframes |
| `--days N` | 60 | days of history |
| `--no-liq` | off | skip 15m/5m liquidations from Coinalyze |
| `--exchanges …` | `Binance Bybit OKX` | which exchanges to collect from |

Writes the same raw tables as `fetch.py` with `interval` = `15min` / `5min`: candles (buy volume on Binance and OKX),
open interest, funding, long/short from the exchanges, and liquidations from Coinalyze (exchanges keep no
liquidation history), with their coverage start in `liquidation_coverage`. Binance OI and long/short go back
30 days, Bybit the full 60. Public endpoints, no key; ~2,000 requests for 10 tickers, markets in parallel.
Hourly data is untouched. Markets come from the `markets` table, so run `fetch.py` for a ticker first.

### `load_pumps.py` — labels into the database

No options. Validates `pumps.csv` (types, order, overlaps) and rebuilds the `pumps` table from it. `label.py`
does this automatically on every change.

### `features.py` — features table

| Option | Default | Meaning |
|---|---|---|
| `tickers …` | all of `tickers.txt` | recompute only these tickers; without arguments the table is rebuilt |

Run after new data or changed labels. Tickers in `tickers.txt` (reviewed) contribute every hour; any other ticker
with labels contributes only the hours inside its labeled intervals (column `reviewed` = false).

## Model

### `train.py` — validation and models for the screener

| Option | Default | Meaning |
|---|---|---|
| `--models …` | `rule logreg lightgbm` | which models to validate |
| `--no-save` | off | report only; do not overwrite `models/*.joblib` |
| `--no-15m` | off | skip the 15m model (faster: hourly models only) |
| `--candidate` | off | save to `models/candidate/` instead of `models/`; the labeling tool compares it with the live model and **Прийняти** promotes it (the live one goes to `models/backup_<time>/`) |
| `--fresh-hours N` | 48 | unlabeled hours from the last N hours are left out of training; `0` uses everything |
| `--with-liq` | off | include the liquidation features (`liq_*`) as before 2026-10-07; by default the models train without them (`features.MODEL_FEATURES`, `MODEL_FEATURES_15M`; see `experiments/no_liq.py`) |
| `--no-move` | off | train the 15m model without the own-move features (`beta`, `idio_*`, `illiq_x`, `park_*`… — as before 2026-10-08); by default they are in (`features.MOVE_FEATURES`, needs the `alt_index` table) |

**Why `--fresh-hours`.** Tickers in `tickers.txt` are treated as fully labeled, so an unlabeled hour counts as
"no pump". Fresh hours nobody has reviewed yet — possibly a pump in progress — would teach the model to un-see
it. Hours inside labeled intervals are always kept. The value used is saved in the model file, shown in the
training report ("Training settings") and in the dashboard's system status.

Outputs: `reports/train_report.md` (with a "15m model and signal tiers" section), `reports/oof_predictions.csv`
(used by the labeling tool's error review), `models/logreg.joblib`, `models/lightgbm.joblib`,
`models/lightgbm_15m.joblib` (15m model, confirm rule, 15m stop multiplier).

### `experiments/oi_coins.py`

No options. Re-runs the OI-in-coins vs OI-in-USD comparison → `reports/experiment_oi_coins.md`.

### `experiments/exchange_direct.py`

| Option | Default | Meaning |
|---|---|---|
| `--refresh` | off | download from Binance and Bybit again instead of using `experiments/cache/` |

Compares Binance/Bybit public APIs with Coinalyze: raw series, features, and model quality →
`reports/experiment_exchange_direct.md`. Does not touch the database.

### `experiments/volume_delta.py`

| Option | Default | Meaning |
|---|---|---|
| `--refresh` | off | download Binance spot candles again instead of using `experiments/cache/` |

Tests perp and spot volume delta features against the current set → `reports/experiment_volume_delta.md`.

### `experiments/m15_model.py`

No options. 15m model on database data (exchange candles/OI + Coinalyze 15m liquidations): base features vs
+liquidations vs +short features vs both, with the current hourly model as reference → `reports/experiment_m15_model.md`.

### `experiments/stop_from_entry.py`

No options. Out-of-fold drawdowns from the 15m model's entries into labeled pumps and a stop-multiplier sweep →
`reports/experiment_stop.md`.

### `experiments/early_phase.py`

Early-phase features for the 15m model: the current set versus the set without "already rose" features (ret_24h,
ret_4h, oi_chg_24h, oi_minus_ret_24h) and with new early features (OI and volume acceleration, share of bars with OI
rising, OI versus price over the last hour, volatility squeeze, buy-share shift). Main metric: the share of the pump's
move already done at the first signal.

### `experiments/exit_vwap.py`

Exit rule for the model's position on out-of-fold entries of the 15m tier (same entries and stop for every variant):
the current "Alert gone for 4 bars" against exits on VWAP (anchored at entry, rolling 1 h or 4 h; 1–3 confirming
closes; 0 or 0.5 ATR buffer; checked on 15m or 5m closes), alone, whichever comes first, or only when both agree.
0.1% fee per trade. Writes `reports/exit_vwap.csv`.

### `experiments/exit_giveback.py`

Exit by the labeling logic: the price gave back a share (¼, ⅓, ½) of the move since entry (move = highest high since
entry minus the entry price, applied once the move is 1–2 ATR), optionally OI −10%, alone, whichever comes first with the
current "Alert gone for 4 bars", or only together with it (`and`, and `and_small`: while the move is small, exit as now).
Out-of-fold Alert entries, k·ATR stop, 0.1% fee. Writes `reports/exit_giveback.csv`.

### `review.py`

| Option | Meaning |
|---|---|
| `--rebuild` | rebuild the blind-review pool (`reports/review_pool.json`) from all backtest runs and the screener journal |
| `--summary` | print the summary of decisions in `reviews.csv` |

The review page itself is served by `label.py` at `/review` (see [labeling.md](labeling.md)).

### `my_trades.py`

| Option | Meaning |
|---|---|
| `--check` | show the API key permissions (must be read-only) |
| `--fetch` | download Binance USDⓈ-M trades (`--days`, default 180) → fills and positions with tilt flags in `reports/my_trades/` (not in git) |
| `--update` | only new trades: from the last saved fill (minus a day), merged with the saved ones — about a minute instead of hours |

The key lives in `.env` (`BINANCE_API_KEY`, `BINANCE_API_SECRET`) and must be read-only; it is never written anywhere else.

### `experiments/my_entries_profile.py`

The owner's systematic long entries (held ≥ 10 min, no tilt flags) versus the model's backtest entries, on Binance 15m
price and volume only: change over 1h / 4h / 24h before entry, distance to the 24h and 7-day high, share of new 7-day
highs, pullback position within the day's range, 4h volume against its 7-day norm, and MFE / MAE / share of the move
taken. Candles for the owner's trades are cached in `reports/my_trades/_klines15.pkl`.

### `experiments/logreg_15m.py`

Logistic regression against LightGBM in the 15m tier on the same features and data: leave-ticker-out on the training
tickers, a backtest on the backtest coins outside training (Alert with the current hourly confirmation) with the current
and the partial exit, and the trade-outcome target from `experiments/outcome_target.py` (AUC and trades).

### `experiments/oi_coins_15m.py`

The 15m model with open interest in coins (no price effect) instead of USD, and with both sets: leave-ticker-out on the
training tickers and a backtest on the backtest coins outside training with the current and the partial exit, plus
feature importance of the coin variants.

### `experiments/hmm_regimes.py`

Hidden market regimes as features: a Gaussian hidden Markov model (own numpy implementation, diagonal covariance) on
1-hour price and OI change, 4-hour price change, hourly volume against its 7-day median and the buy share, fitted on the
backtest coins before the backtest window only. State probabilities come from the forward pass alone (filtering, no
look-ahead), states ordered from "falling" to "rising". Reports the state profiles, the state probabilities before and at
the start of impulses, and base vs base + state probabilities: leave-ticker-out and a backtest on the backtest coins
(current and partial exit). `--states N` — number of hidden states (default 4). Report: `reports/hmm_regimes.md`.

### `experiments/market_regime.py`

Market regime (risk on / risk off) against the model's trades. Hourly market features known at the hour's close: BTC
24-hour return, distance to its 7-day mean and 7-day high, BTC volatility against its 30-day norm (Binance 1h, cached in
`data/market_btc.pkl`), and the altcoin market from the 205 coins of the long backtest cache (`data/long_bt`): share of
coins up over 24 hours and 7 days, median 24-hour return, alts minus BTC. Each trade gets the last hour that closed
before its entry. Trade sets: the 6-month long backtest (`reports/long_backtest_trades.csv`) and a September backtest
run. Reports terciles per feature with month-by-month stability, a preset risk-on / risk-off combination, and a filter
"skip the Alert when more than 50 / 60 / 70% of alts are up over 7 days". `--bt PATH` — September run (default: the
newest market run with turnover ≥ 20 M); `--refresh-btc` — download BTC again. Report: `reports/market_regime.md`.

### `experiments/no_liq.py`

Do the models still need liquidations on the impulse labels? Trains the 15m and hourly LightGBM with and without the
`liq_*` features on the current labels (same sample rules as `train.py`) and compares: leave-ticker-out (15m Candidate
tier, hourly events) and a backtest on the backtest coins outside training (Alert = 15m + hourly ≥ 0.75 of its threshold,
hourly features from own 15m data as in the screener), current and partial exit. Variants: `base`, `noliq` (both models
without liquidations), `noliq15` (15m without, hourly with). Report: `reports/no_liq.md`.

### `experiments/book_features.py`

Order book depth features from the free Binance `bookDepth` archive (cumulative bid / ask notional within ±1 / 2 / 5% of
the price every ~30–40 s; `orderbook.archive_day`, cache `data/book_cache`). At each 15m bar close: imbalance, ask and
total depth relative to 24-hour turnover, 1- and 4-hour change of ask / bid depth, ask depth against its 7-day median;
plus short dynamics from all snapshots inside the bar (5 / 15-minute change, mean / min / max imbalance, depth
dispersion). Variants `base` (model features without liquidations), `book`, `book_fast`: leave-ticker-out and a backtest
on ≥ 20 M $ coins outside training. Report: `reports/book_features.md` (first pass: `reports/book_features_pass1.md`).

### `experiments/binance_only.py`

The 15m model on Binance data only vs Binance + Bybit (Binance OI backfilled from the archive first): leave-ticker-out
and a backtest on the backtest coins with three arms — both exchanges, 15m Binance-only with the usual hourly
confirmation, everything Binance-only. Report: `reports/binance_only.md`.

### `experiments/long_binance.py`

Six-month backtest (April–September) on the Binance archive, by month — the robustness check for new features. Coins:
those of the long backtest (`data/long_bt`), outside training. Stages, each cached so a rerun continues: `data`
(`binance_archive.collect` into the raw tables, Binance only; the order book archive downloads in parallel), `features`
(15m model features, order book features, hourly features from own 15m, prices → `data/long_bin/<coin>.pkl`), `test`
(15m variants trained on the training tickers, Binance only; Alert with the live hourly model; partial exit, stop
2.48 ATR → `reports/long_binance.md`, `reports/long_binance_trades.csv`). `--stage data|features|test|all`,
`--limit N` (first N coins, to test the pipeline). New feature ideas plug in as new variants and reuse the cached data.

### `experiments/trade_count.py`

The number of trades as features: on 1h from Coinalyze `tx` / `btx` (all exchanges), on 15m from Binance klines (trade
count field, cached in `data/trades15m/`). Features: trade-count z-score and 24-hour ratio, average trade size z-score,
hourly trade burst, and on 1h the share of buy trades. Base vs base + trade features: leave-ticker-out for the hourly and
the 15m model and a backtest of the 15m tier on the backtest coins. `--tf 1h|15m|both`, `--no-backtest`.
Report: `reports/trade_count.md`.

### `experiments/flow_archive.py`

Trade flow and footprint from the Binance trade archive (`data.binance.vision`, daily `aggTrades`) as 15m-model features.
Each day is downloaded into memory and reduced to 15m summaries (aggressor buy / sell USD, trades ≥ $10k and < $2.5k,
largest trade) and a footprint on a 0.1% log-price grid; cached in `data/flow/<coin>/<date>.pkl`. Features: share and
delta of large trades, small-trade delta, largest trade and average order size against the coin's 7-day history;
delta in the top / bottom third of the bar's range, volume share at the top, POC position (bar and last hour). Compares
base / + flow / + footprint / + both by leave-ticker-out and a backtest on the backtest coins, and shows the features on
the first bar of impulses vs failed impulses. `--workers N` parallel downloads (default 8), `--no-backtest`.
Report: `reports/flow_archive.md`.

### `experiments/spring_pattern.py`

The owner's "stop-run below a local low that gets bought back" pattern (a b-shaped bar) and a selling climax followed by a
positive delta, measured on the cached Binance trade archive (`data/flow/`, from `experiments/flow_archive.py`) and
Binance 15m candles: forward price after 1/4/12 hours, a trade with the stop under the bar's low (+1R / +2R before the
stop, average R after fees), the mirror pattern at highs, controls (any pierce, wick, breakdown, random bars), and how
often the pattern precedes labeled impulses. Report: `reports/spring_pattern.md`, events `reports/spring_events_A.csv`.

### `experiments/spring_context.py`

The same stop-run pattern only in the model's context (a Candidate or Alert within the last 6 hours), on the backtest
coins outside training with the working models (`backtest.tiers_and_bars`), against the model's own entry (first Alert
bar, k·ATR stop), a plain wick below the low in context, the pattern outside context and random context bars; also with
a next-bar confirmation and split into two halves of the period. `--days N` (default 40). Report:
`reports/spring_context.md`, events `reports/spring_context_events.csv`.

### `experiments/entry_model.py`

The "entry model": a small LightGBM trained only on bars in the model's context (Candidate / Alert within 6 h), target
"+2R before a stop under this bar's low (at least 0.5 ATR) within 12 h", footprint, trade-flow and model-state features;
trained on one half of the coins in the first half of the period, tested on the other coins in the second half (and
vice versa), trades one at a time per coin, against rule b, the main model's entry and random context bars. The frame
is cached in `reports/entry_model_frame.pkl`. `--days N` (default 40). Report: `reports/entry_model.md`.

### `experiments/feature_corr.py`

Spearman correlation between the 34 features of the 15m model on 15m bars of all coins (the cache of
`experiments/outcome_target.py`): the strongest pairs and the most independent features in the console, and a heatmap
ordered by hierarchical clustering in `reports/feature_corr.html`.

### `experiments/limit_entry.py`

A limit buy on a pullback after the Alert instead of a market buy at the signal bar's close: at the VWAP of the last 1–2
hours or at close − 0.5 / 1 ATR, valid 1 or 2 hours; then the current position rule (partial exit, breakeven, model exit).
Reports fill rate, PF overall and by turnover band, how many ≥ +10% trades were missed, and waiting time. Uses the caches
of `experiments/exit_structure.py` (Alert scores) and `experiments/style_backtest.py` (Binance 15m bars with volume).

### `strategies.py`

Modes: a model plus trade rules — signal (Alert or Candidate), entry (market or a limit at −0.5 ATR, valid 1 hour), the
order book filter (bid / ask ratio ≥ 1 at the signal) — with the shared partial exit. The list lives in
`strategies.json` (a new experiment is a new entry); `python strategies.py` prints it. The same mode selector drives the
dashboard's Forecast (position, trade marks), Journal (statistics per mode) and Backtest (`backtest.py --strategy <id>`;
the order book filter reads the Binance `bookDepth` archive there).

Mode `p_be` ("б/з на p", key `p_be: true`): a footprint **p** mark (absorption above the highs, `footprint.py`) inside an
open long moves the stop to breakeven, and after such a trade there is no re-entry until the Alert episode ends
(`risk.position_state(p_marks=)`; marks from `fp_marks`, so it only applies where the footprint is recorded — live and
the 30-day history of list coins; for history use `experiments/p_exit.py`).

Mode `market_cool` ("ринок не гарячий", key `market_max: 0.6`) takes an Alert only when at most 60% of altcoins are up
over 7 days at the close of the signal bar (`market.py`; unknown state — the signal is taken).

### `market.py` — altcoin market state

| Usage | Meaning |
|---|---|
| `python market.py` | update `market_breadth`: hourly closes of ~250 liquid Binance alts (30-day average turnover 5 M – 1 B $, no stablecoins) and BTC → share of alts up over 24 h / 7 days, median alt move, alts vs BTC, BTC trend and volatility. First run downloads 60 days (~2 min, rate-limited to ~900 weight/min); later runs fetch only new hours. Closes are cached in `reports/market_closes.pkl` |
| `python market.py --show` | print the latest state, download nothing |

State: **Hot** — more than 60% of alts up over 7 days, **Cold** — below 40%, otherwise Neutral. The screener (`--loop`)
updates it in its own thread at HH:01; the dashboard reads it at `/api/market`; journal, dashboard positions and
`backtest.py` use it for the `market_cool` mode. Background: `experiments/market_regime.py`.

### `movers.py` — coins with idiosyncratic moves

| Usage | Meaning |
|---|---|
| `python movers.py [--h 1\|4\|24]` | compute from the `market.py` cache and print the top: beta and R² vs the equal-weighted alt index (OLS on hourly log returns, 7 days ending 24 h ago), idiosyncratic return (total − β × index), residual z-score, Amihud illiquidity vs its 30-day median, 24 h turnover; writes `reports/movers.json` |

The screener's market thread runs it every hour after `market.update` (which now also caches hourly quote volume in
`reports/market_qvol.pkl`). Dashboard: the **Movers N ▾** button (`/api/movers`); N = coins with |z| ≥ 2.5. Owner's
guide to every column: `notes/movers.md`.

### `drivers.py` — why the model gave this probability

| Usage | Meaning |
|---|---|
| `python drivers.py NMR` | contributions of every feature to the 15m model's P on the last scored bar: top features pushing P up and pulling it down (value now and share of all pushes), per-group shares, and the one-line Telegram text |
| `python drivers.py NMR --ts "2026-10-07 11:00"` | the same for a given 15m bar (open time, UTC) |

LightGBM decomposes each prediction exactly (TreeSHAP, `booster.predict(..., pred_contrib=True)`): log-odds = base value +
sum of contributions. The share of a contribution in the sum of all |contributions| is shown instead of a probability
delta, because near P ≈ 1 removing one feature hardly moves P. Features are built as the screener builds them
(`features.build_15m`, 16 days of history and the ALT-EW index), so P matches the stored score of the same model version.
A warning `No volume behind this signal` is added when P is above the threshold and the Volume group pulls P down or adds
nothing. The dashboard (`/api/ticker` → `drivers`) and the Telegram bot use the same function.

### `monitor.py` — live health of the 15m model

| Usage | Meaning |
|---|---|
| `python monitor.py [--days 7]` | report on the current 15m model version, out-of-training bars of the last N days → `reports/monitor.md` + `reports/monitor.json`: (1) signal frequency per day (tickers scored, Candidate / Alert bars, tickers with an Alert); (2) calibration — bars grouped by `s15 = P / threshold` with what the price did next: median return after 1 / 4 h, max rise / drawdown over 4 h in ATR15 (MFE / MAE), share of bars followed by a ≥ 2 ATR rise, share inside labeled impulses; (3) feature drift — PSI of every model feature, list tickers vs the training sample, with medians train → now |
| `python monitor.py --add-ref` | write the training feature distribution (`feature_ref`: deciles, NaN share, median) into the live 15m model; models trained after 2026-10-08 get it from `train.py` automatically |

PSI < 0.1 — stable, 0.1–0.25 — noticeable shift, > 0.25 — strong shift (the model sees values it rarely saw in training).
The watchlist is not a random sample (the scanner adds coins that are waking up), so some shift in volume / volatility is
expected from selection alone. Hourly confirmation for the Alert tier uses the latest hourly score of each hour across
model versions. Run it on the server, where the live scores are:
`docker compose -f docker-compose.server.yml exec -T screener python -u monitor.py`.

### `alt_index.py` — ALT-EW index for the own-move features

| Usage | Meaning |
|---|---|
| `python alt_index.py` | bring the `alt_index` table up to the last closed 5m bar: equal-weighted mean 5m log return of the same ~250 alts as `market.py`. Empty table — history from 2026-06-01 out of the data.binance.vision archive (monthly / daily 5m klines, not cached on disk; ~6 min); a gap longer than ~3.5 days — full days from the archive, the rest from REST |
| `python alt_index.py --from 2026-05-01` | rebuild the history from that date to yesterday (overwrites those bars) |
| `python alt_index.py --show` | print the coverage only |

The screener (`--loop`) updates it in its own thread 8 s after every 5m bar closes; before scoring a 15m bar (or a rolling
one) it checks the bar is there (`alt_index.ensure`, REST tail only). A 15m model trained with the own-move features
needs this table — on a new machine or server run `python alt_index.py` once before `train.py` / the screener.

### `orderbook.py`

| Usage | Meaning |
|---|---|
| `python orderbook.py GTC MOVR` | print the live bid / ask ratio within ±2% of the price (Binance + Bybit), per exchange too; nothing is written |

The screener records snapshots for coins near a signal into `book_ratio` (every cycle); the dashboard reads the live value
(`/api/book`) and the value at each Alert.

### `wake.py`

| Usage | Meaning |
|---|---|
| `python wake.py AIN --days 3` | "waking up" events of a coin (5m: 15-min volume ≥ 5× norm, price +2%, 1 h OI +2%) from the database; the screener runs the same rule every cycle → `reports/wake_events.json` |

### `volume_profile.py`

| Usage | Meaning |
|---|---|
| `python volume_profile.py SOON --days 2` | daily volume profiles (UTC days) for the last N days: POC, value area, step, volume |
| `python volume_profile.py SOON --from "2026-10-02 08:00" --to "2026-10-02 14:00"` | one profile for a fixed range |

Same code serves `/api/profile` in the dashboard and the labeling tool (`app/js/profile.js`). Built from candles, not
ticks: each candle's volume (Binance + Bybit, USD) is spread evenly over its low–high range; the finest timeframe that
covers the range is used (5m → 15m → 1h). Value area = 70% of volume, grown from the POC towards the larger neighbour.

### `experiments/klines1m.py`, `experiments/l_shock.py`, `experiments/profile_levels.py`

| Usage | Meaning |
|---|---|
| `python experiments/klines1m.py` | 1m Binance USDT-perp candles from data.binance.vision (Apr–Sep 2026) → `data/klines1m/<coin>.parquet`; reruns fetch only missing months |
| `python experiments/klines1m.py --months 2026-10 SOON` | given months / coins |
| `python experiments/l_shock.py` | "liquidation fade" event test on 5m and 1m with controls → `reports/l_shock.md`, `reports/l_shock_trades.csv` (`--no-oi` skips the database) |
| `python experiments/profile_levels.py` | previous-day POC / VAH / VAL and impulse POC vs random levels → `reports/profile_levels.md` |

### `experiments/time_of_day.py`

| Usage | Meaning |
|---|---|
| `python experiments/time_of_day.py` | trades of the current rule by hour of day (UTC and Kyiv), 4-hour blocks and weekday: 6-month cache + September with the current model → `reports/time_of_day.md` |

### `experiments/wakeup_5m.py`

| Usage | Meaning |
|---|---|
| `python experiments/wakeup_5m.py` | "waking up" rule on 5m bars (15-min volume × norm, 15-min price change, optional 1 h OI) on ≥ 20 M$ backtest coins in September: how often a model Candidate / Alert follows, how much earlier, what price does next, against random moments → `reports/wakeup_5m.md`, `.csv` (model tiers cached in `reports/backtests/_wakeup_tiers.pkl`) |

### `experiments/turnover_at_entry.py`

| Usage | Meaning |
|---|---|
| `python experiments/turnover_at_entry.py` | backtest trades (Oct 3 runs) vs the coin's 24 h Binance turnover at entry and its 30-day average → `reports/turnover_at_entry.md`, `.csv` |

### `experiments/pullback_grid.py`

| Usage | Meaning |
|---|---|
| `python experiments/pullback_grid.py` | pullback (limit) entry after Alert: depth (market, −x ATR, −y%) × order validity (1/2/4 h) × stop (fixed % or k·ATR) on the 6-month Bybit cache (choose on Apr–Jun, check on Jul–Sep) and the September Binance cache → `reports/pullback_grid.md`, `reports/pullback_grid.csv` |

### `experiments/long_backtest.py`

The entry rules (market, order book filter OB ≥ 1, limit −0.5 ATR, both) over 6 months on Bybit data (15m candles and
open interest; Binance keeps only ~30 days of 15m OI), with no liquidations and the Binance `bookDepth` archive for the
order book. Stages are cached (Bybit data in the database, model scores in `data/long_bt/`, order book days in
`data/book_cache/`); `--skip-fetch` skips the download, `--limit N` runs on N coins. Reports PF overall, by turnover band
and by month; trades go to `reports/long_backtest_trades.csv`.

### `experiments/entry_rules_backtest.py`

The two entry rules that passed the stability checks — the ±2% order book imbalance filter (bids ≥ asks) and a limit
entry at close − 0.5 ATR valid one hour — separately and together, against the current market entry, on the backtest
coins outside training with the partial exit. The imbalance is taken at every signal from the Binance `bookDepth`
archive. Reports PF by turnover band and by half of the period; trades go to `reports/entry_rules_trades.csv`.

### `experiments/book_filter.py`

A filter model on all order book and spot features at entry (`reports/book_spot_features.csv` from
`experiments/book_spot.py`): logistic regression and a shallow LightGBM, trained on one half of the period and tested on
the other in both directions, against all trades and the single "±2% book imbalance" rule.

### `experiments/exit_partial.py`

Partial exit for the current model's Alert entries: close ⅓ or ½ of the position at +2 / +3 / +4 ATR, keep the rest on
the current exit (4 bars without Alert or the stop), with the stop either unchanged or moved to breakeven after the
partial exit; stop 2 or 2.5 ATR. Grid on the backtest coins by turnover band, then the best set against the current exit
on the screener journal.

### `experiments/exit_barrier_grid.py`

Robustness of a barrier exit (target +TP·ATR, stop −SL·ATR, time limit) for the current model's Alert entries: a grid of
neighbouring parameters on the backtest coins (from the `experiments/exit_structure.py` cache), split by turnover band,
and the same comparison on the screener journal's out-of-sample trades.

### `experiments/outcome_target.py`

A 15m model trained on the trade outcome instead of "impulse is running": triple-barrier labels (+3 ATR before −1.5 ATR
within 4 hours) on "live" moments (4h volume ≥ 2× its norm or a 2% move in an hour) across all coins with 15m data,
validated out of sample in both coins and time (group A early → group B late and the reverse; thresholds are quantiles
of the training predictions). Reports test AUC, non-overlapping trades at several thresholds against all live moments
and the current model, and feature importance. Features are cached in `reports/backtests/_outcome_feats.pkl`
(`--refresh` rebuilds).

### `experiments/level_features.py`

Level features for the 15m model (distance to the 24h and 7-day high, in % and in ATR, position in the day's range,
break of the 7-day high): two identically retrained models, with and without them, compared by leave-ticker-out on the
training tickers and by a backtest on the backtest coins outside training with the dashboard's position logic.

### `experiments/style_backtest.py`

A mechanical version of the owner's entry style on the backtest coins: a hot coin (24h change and 4h volume against
its 7-day norm) that has pulled back 3–10% from its 24h high, against "chasing" at the high and random entries, with
one shared exit (3% stop or 4 hours) and forward prices after 1 / 4 / 12 hours; each rule is also split by whether the
model's Alert was on at entry. Uses the model-score cache of `experiments/exit_structure.py`.

### `experiments/exit_structure.py`

Exit rules from the owner's review of two stopped trades, on the backtest coins (outside training) with the full
position logic: the current k·ATR stop, an early exit after 2 or 3 red 15m bars with falling OI and no new high, a stop
just below the entry bar's low, and their combinations. Reports win rate, mean, median, sum without the top 3, profit
factor, stop share, holding time and how many ≥ +5% winners each rule cuts. Model scores are cached in
`reports/backtests/_exit_structure_cache.pkl` (`--refresh` recomputes).

### `experiments/in_play.py`

Backtest trades (coins outside training) split by whether the coin was "in play" before the entry: 24h turnover ×N its
30-day median, OI up over 3 days, 24h range ×N its median, or a top mover of the hour among all backtest coins, at any
time in the 5 days before entry (with and without the last 6 hours), at three threshold sets. Also price excursions
after model entries versus random entries on the same coins.

### `experiments/early_5m.py`

Earlier signals: the same 15m model on rolling 15m bars every 5 minutes (from 5m exchange data and 5m liquidations)
versus the aligned :00/:15/:30/:45 bars, Candidate on both, and an early "Підготовка" tier (15m score ≥ 0.93 twice with
price and OI rising): caught impulses, delay to the close of the first signal bar, share of the move done, failed
impulses with a signal, false alarms per 30 days. Result: rolling bars are now live in `screener.py --loop` (Alert
−5 min, no extra false alarms); the early tier was rejected (7.8 false alarms per 30 days). `experiments/backtest_slices.py` splits a backtest run by scanner
filter and turnover band.

### `experiments/flush_risk.py`

Long flush after a stall: in the 12 hours after an Alert, how often price drops 3+ ATR(15m) within 4 hours when it
stalls near the 24-hour high, with open interest at its high, the 15m score fading, a high long share or positive funding
(one situation per 4 hours per ticker).

### `experiments/in_position.py`

Signals inside the model's open position: what price does 1/2/4 hours after a bar where the Alert keeps burning, flares
up again after a 1–3-bar gap, or the 15m score falls three bars in a row with no new high for four bars ("fading"),
compared with the first entry and other bars; plus trades with an extra "fading" exit (medians, and the sum without the
two largest trades).

### `experiments/exit_examples.py`

Visual check of the giveback exit: an SVG page `reports/exit_rule_examples.html` with the trades where the current and
the new exit differ (candles, Alert strip, entry, both exits, stop, the half-of-the-move level).

### `experiments/scale_in.py`

Entering in parts: a fraction of the position on the Candidate, the rest added on the Alert, against all-in on the Alert
and all-in on the Candidate. One stop for the whole position from the first entry, 0.1% fee per part, results in units
of a full position; split by trades inside labeled impulses and outside them.

### `experiments/rolling_confirm.py`

Faster Alert? The hourly confirmation on a rolling hour (updated every 15 minutes from our 15m data) against the closed
calendar hour, from Coinalyze or our data, at confirm share 0.75 or 0.6, and Candidate alone: caught impulses, delay,
share of the move done at the first signal, failed impulses with a signal, false alarms per 30 days.

### `experiments/hourly_from_15m.py`

Hourly context (Alert confirmation) from our own 15m data (Binance + Bybit + Coinalyze 15m liquidations, rolled up to
hours) versus Coinalyze hourly data: feature correlations, hourly-model P, agreement of the confirmation decision and
Alert events on the 15m grid. `--days N` — compare the last N days (default 60).

### `experiments/timeframe.py`

| Option | Default | Meaning |
|---|---|---|
| `--tf 15m\|5m` | `15m` | smaller timeframe to compare with 1h |
| `--refresh` | off | download from Binance and Bybit again instead of using `experiments/cache/` |

Same features and labels on 1h and on 15m/5m bars from the exchanges; compares detection delay, the share of
the move already gone at the signal, and false alarms → `reports/experiment_timeframe_<tf>.md`.

## Live

### `screener.py` — hourly scoring of `watchlist.txt`

| Option | Default | Meaning |
|---|---|---|
| `--loop` | off | run forever: every 15 minutes (HH:02/17/32/47 UTC) the 15m tier, every 5 minutes in between the "hot" coins on rolling 15m bars, once an hour also the hourly models |
| `--no-fetch` | off | score data already in the database, no API calls (quick check, or re-score after retraining) |
| `--no-scan` | off | skip the market scan (every 15 minutes); use `watchlist.txt` and the existing auto-list only |

Each cycle scores `watchlist.txt` plus the scanner's live auto-list (`reports/auto_watchlist.json`); both are re-read
every cycle, so new tickers need no restart. Writes the `signals` table and `reports/screener_status.json` (shown in the dashboard). Loads models at start:
restart it after `train.py`.

### `scanner.py` — market scan for the auto-list

| Option | Meaning |
|---|---|
| (none) | scan all Binance USDT perps now and print who passes each preset and the WOKE channel; the auto-list is not changed |
| `--apply` | also update `reports/auto_watchlist.json` (what `screener.py` does every 15 minutes) |
| `--recall` | test every preset on the labeled pumps from the database; writes `reports/scan_recall.json`, shown in the dashboard |

Thresholds live in `scan_config.json` (edited from the dashboard). See [screener.md](screener.md#market-scanner).

### `backtest.py` — wide backtest on coins outside training

| Option | Default | Meaning |
|---|---|---|
| `--min-turnover` | 20e6 | average daily turnover over the window, $ |
| `--max-oi` | scanner's `max_oi` (500 M$) | open interest cap — same scope as the scanner (no BTC/ETH) |
| `--days` | 30 | window (Binance keeps open interest history for 30 days) |
| `--model` | current | `current` or `candidate` (models/candidate/) |
| `--entry` | alert | `alert` or `candidate` (15m model alone) |
| `--strategy` | — | a mode from `strategies.json` (`market`, `limit`, `ob`, `ob_limit`, `cand`): signal, entry and order book filter; overrides `--entry` |
| `--k` | — | stop in ATR instead of the one saved with the model (compare models at the same stop) |
| `--no-fetch` | off | use only data already in the database |
| `--limit` | 0 | only the first N coins (a quick check) |

The coin list is fixed by rule before the run: Binance USDT perps by turnover and open interest, minus `tickers.txt`,
labeled coins and `excluded.txt`. Data: 15m from Binance and Bybit plus Coinalyze 15m liquidations (~3 calls per coin;
contracts for all coins in one call); the hourly confirmation comes from our own 15m data. Trades follow the position
rules (`risk.position_state`), 0.1% fee. Writes `reports/backtests/<time>_<model>_<entry>.json`; the dashboard tab
**Бектест** shows and launches runs. The first run on ~100 coins takes 30–60 minutes (data), later ones only top up.

### `dashboard.py` — Forecast and Journal

| Option | Default | Meaning |
|---|---|---|
| `--port N` | 8766 | port on 127.0.0.1 |
| `--no-browser` | off | do not open the browser |
| `--host ADDR` | 127.0.0.1 | address to listen on (0.0.0.0 in the server container) |
| `--quick-add TICKER` | — | only the quick part of adding a ticker (exchange contracts, 15m/5m from the exchanges, the 15m and own-hourly score) and exit; the dashboard runs this itself as a separate low-priority process after "+ to watchlist", so charts are not slowed down. Coinalyze history is collected later by the screener |

`http://127.0.0.1:8766/#forecast/NMR` opens a ticker, `#journal` the journal. Loads models at start: restart it
after `train.py`.

### `label.py` — labeling tool

| Option | Default | Meaning |
|---|---|---|
| `--days N` | 60 | days of history to show |
| `--port N` | 8765 | port on 127.0.0.1 |

The page has a 1h / 15m / 5m switch; 15m and 5m need `fetch_exchange.py` for that ticker.
| `--no-browser` | off | do not open the browser |

`http://127.0.0.1:8765/#AKE` opens a ticker. If the port is taken, the script says so and suggests another one.

## Infrastructure

| Command | Meaning |
|---|---|
| `docker compose up -d` | start PostgreSQL |
| `docker compose ps` | status (should be healthy) |
| `docker compose down` | stop; data is kept |
| `docker compose exec db psql -U postgres -d pump_detector` | SQL console |

### experiments/wake_channel.py

Scanner channel "woke up": coins whose 30-day average turnover is below $20M but which are liquid right now (4h turnover × 6 ≥ $20M) and spike (4h volume ×N vs the 7-day norm, price and OI up over 4h). Tests five threshold variants on the last ~30 days of all Binance USDT perps (15m klines + OI via REST, cache `data/wake_channel/`) against random quiet bars → `reports/wake_channel.md`, events in `reports/wake_channel_events.csv`.

```
python experiments/wake_channel.py          # whole market, last ~30 days
python experiments/wake_channel.py --owner  # + does the rule fire before the owner's trades on quiet coins (OI from the Binance archive)
```

### experiments/trade_patterns.py

What separates the model's winning trades from losing ones: features at entry (exchange trade count vs norm, NATR and NATR vs
its 7-day norm, range, returns, distance from the 24h high, taker-buy share, turnover, hour, a loss on the same coin in the
previous 6 h) for the trades of `reports/long_binance_trades.csv` (6 months, Binance archive, honest universe), computed from
the monthly 15m klines cache. Quintile PF in April–June and July–September separately → `reports/trade_patterns.md`,
per-trade features → `reports/trade_patterns.csv`.

```
python experiments/trade_patterns.py
```

### experiments/liq_map.py

Estimated liquidation map (like Coinglass / Hyblock heatmaps) from the Binance archive at 5 minutes: OI growth split into new
longs / shorts by the bar's taker-buy share, leverage 10 / 25 / 50× in equal parts, a level burns when price reaches it, the
map shrinks with falling OI and decays with a 3-day half-life. At the entry of each trade in `reports/trade_patterns.csv`:
short fuel 0–3% / 3–8% above price, long fuel below, their ratio, breakout of the 24h / 7-day high. Quintile PF in two halves
→ `reports/liq_map.md`, per-trade features → `reports/liq_map.csv`.

```
python experiments/liq_map.py [--workers 6]
```

