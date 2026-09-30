# Commands

Every script in the project with its options and when to use them. `python <script> --help` prints the same
options straight from the code. Run everything from the project folder with the virtual environment active.

## Typical workflows

| Task | Commands |
|---|---|
| Add a ticker for training | in `python label.py`: find it under **Усі Binance** → **Завантажити** → label it → after reviewing the whole ticker press **У навчання** → `python features.py` → `python train.py` → restart `screener.py` and `dashboard.py` |
| Confirm a journal signal | open the ticker in `label.py`, label the interval as `pump` or `false_start` → `python features.py` (only that interval enters training) |
| Scan a coin live only | add it to `watchlist.txt`; the screener picks it up on its next cycle (7 days of history needed) |
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
| `--loop` | off | run forever: every 15 minutes (HH:02/17/32/47 UTC) the 15m tier, once an hour also the hourly models |
| `--no-fetch` | off | score data already in the database, no API calls (quick check, or re-score after retraining) |

Writes the `signals` table and `reports/screener_status.json` (shown in the dashboard). Loads models at start:
restart it after `train.py`.

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
