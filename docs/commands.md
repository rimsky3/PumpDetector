# Commands

Every script in the project with its options and when to use them. `python <script> --help` prints the same
options straight from the code. Run everything from the project folder with the virtual environment active.

## Typical workflows

| Task | Commands |
|---|---|
| Add a ticker for training | add it to `tickers.txt` → `python fetch.py --new` → label it in `python label.py` → `python features.py` → `python train.py` → restart `screener.py` and `dashboard.py` |
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

### `load_pumps.py` — labels into the database

No options. Validates `pumps.csv` (types, order, overlaps) and rebuilds the `pumps` table from it. `label.py`
does this automatically on every change.

### `features.py` — features table

| Option | Default | Meaning |
|---|---|---|
| `tickers …` | all of `tickers.txt` | recompute only these tickers; without arguments the table is rebuilt |

Run after new data or changed labels. Only `tickers.txt` is used: watchlist-only coins are unlabeled and must
not enter training.

## Model

### `train.py` — validation and models for the screener

| Option | Default | Meaning |
|---|---|---|
| `--models …` | `rule logreg lightgbm` | which models to validate |
| `--no-save` | off | report only; do not overwrite `models/*.joblib` |
| `--fresh-hours N` | 48 | unlabeled hours from the last N hours are left out of training; `0` uses everything |

**Why `--fresh-hours`.** Tickers in `tickers.txt` are treated as fully labeled, so an unlabeled hour counts as
"no pump". Fresh hours nobody has reviewed yet — possibly a pump in progress — would teach the model to un-see
it. Hours inside labeled intervals are always kept. The value used is saved in the model file, shown in the
training report ("Training settings") and in the dashboard's system status.

Outputs: `reports/train_report.md`, `reports/oof_predictions.csv` (used by the labeling tool's error review),
`models/logreg.joblib`, `models/lightgbm.joblib`.

### `experiments/oi_coins.py`

No options. Re-runs the OI-in-coins vs OI-in-USD comparison → `reports/experiment_oi_coins.md`.

## Live

### `screener.py` — hourly scoring of `watchlist.txt`

| Option | Default | Meaning |
|---|---|---|
| `--loop` | off | run forever, one cycle every hour at HH:05 UTC |
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
| `--no-browser` | off | do not open the browser |

`http://127.0.0.1:8765/#AKE` opens a ticker. If the port is taken, the script says so and suggests another one.

## Infrastructure

| Command | Meaning |
|---|---|
| `docker compose up -d` | start PostgreSQL |
| `docker compose ps` | status (should be healthy) |
| `docker compose down` | stop; data is kept |
| `docker compose exec db psql -U postgres -d pump_detector` | SQL console |
