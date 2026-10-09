# Setup

## Requirements

- Python 3.10+ (developed on 3.14)
- Docker Desktop (runs PostgreSQL 17)
- A [Coinalyze](https://coinalyze.net) API key (free)

### Windows notes

Docker Desktop needs WSL2. If Docker reports *"Virtualization support not detected"* while
virtualization is enabled in BIOS, WSL is simply not installed yet. In an administrator PowerShell:

```powershell
wsl --install --no-distribution
```

Reboot, then start Docker Desktop. If the install hangs on downloading, add `--web-download`.

Docker keeps its data in a single virtual disk that grows over time. If the system drive is small,
move it: Docker Desktop → Settings → Resources → Advanced → **Disk image location**.

## Configuration

Copy `.env.example` to `.env` and fill it in:

```ini
COINALYZE_API_KEY=your_key_here
POSTGRES_PASSWORD=any_password_you_choose
DATABASE_URL=postgresql://postgres:any_password_you_choose@localhost:5432/pump_detector
```

`POSTGRES_PASSWORD` is not issued by anyone: you choose it, and Docker sets it when it creates the
database. The same password goes into `DATABASE_URL`. `.env` is git-ignored and must never be committed.

## Database

```bash
docker compose up -d        # start (first start creates the database and applies schema.sql)
docker compose ps           # status should be "healthy"
docker compose down         # stop, data is kept in the pgdata volume
```

`schema.sql` runs automatically **only when the data volume is empty**. To apply schema changes
to an existing database, run the SQL manually:

```bash
docker compose exec -T db psql -U postgres -d pump_detector < schema.sql
```

To start over from an empty database (deletes all collected data):

```bash
docker compose down -v && docker compose up -d
```

## Python environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Everyday commands

Full reference with every option: [commands.md](commands.md).

| Command | What it does |
|---|---|
| `python fetch.py` | Collects 60 days of hourly data for **every** ticker in `tickers.txt` |
| `python fetch.py --new` | Collects only tickers from `tickers.txt` that are not in the database yet or lack the full hourly history |
| `python fetch.py AKE QNT` | Collects only the given tickers |
| `python fetch.py --days 30` | Changes the history depth |
| `python label.py` | Opens the labeling tool at http://127.0.0.1:8765 |
| `python load_pumps.py` | Validates `pumps.csv` and reloads it into the `pumps` table |

Re-running `fetch.py` is safe: rows are upserted, never duplicated.

### Adding a ticker vs refreshing everything

The API limit is counted **per symbol per request**, not per row, so fetching fewer days does not make a
run cheaper. The cost is driven by the number of markets: roughly 3 markets per ticker × 5 series.
A full run over 7 tickers is ~126 calls, about 3–4 minutes at 40 calls/min, and grows with the list.

| Situation | Command |
|---|---|
| Added a ticker to `tickers.txt` | `python fetch.py --new`, which takes seconds and fetches only the new ones |
| Regular refresh (for example weekly) | `python fetch.py`, which also appends the latest hours for existing tickers |

The provider keeps only ~2 months of hourly history, so the regular full refresh is what extends the
dataset over time. A ticker counts as "in the database" once its markets are stored; a ticker that was
not found on any exchange is retried on every `--new` run.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `429 від Coinalyze: чекаю N с…` | Rate limit of 40 calls/min was hit, usually because of another run in the same minute. The script waits and continues. |
| `Помилка 401` | Wrong or missing `COINALYZE_API_KEY` in `.env`. |
| `There is not enough space on the disk` | The system drive is full. Free space or move Docker's disk image (see Windows notes). |
| `connection refused` on port 5432 | The database is not running: `docker compose up -d`. |
| `label.py` says the address is in use | Another instance is running. Stop it, or use `python label.py --port 8766`. |
| `LF will be replaced by CRLF` from git | Harmless line-ending warning on Windows. |
