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
| Daily operation | `python screener.py --loop` and `python dashboard.py` in two terminals |

Before retraining, label fresh signals on training tickers if they were pumps (see `--fresh-hours`).

## Data

### `fetch.py` — Coinalyze → PostgreSQL

| Option | Default | Meaning |
|---|---|---|
| `tickers …` | all of `tickers.txt` | base assets to collect, e.g. `AKE QNT` |
| `--days N` | 60 | days of hourly history to request |
| `--new` | off | only tickers from `tickers.txt` that are not in the database yet (fast) |

Re-running is safe: rows are upserted. The API limit counts symbols, not days, so `--days` does not make a run
cheaper; `--new` does.

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
| (none) | scan all Binance USDT perps now and print who passes each preset; the auto-list is not changed |
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
