# Server deployment

Since 2026-10-06 the live part of the system (database, screener, dashboard) runs on a small cloud server 24/7;
labeling, training, backtests and experiments stay on the laptop.

## Host

| | |
|---|---|
| Provider | AWS Lightsail, Stockholm (eu-north-1) — not a US region: Binance and Bybit block US IPs |
| Plan | 2 vCPU, 2 GB RAM (+ 2 GB swap), 60 GB SSD, daily automatic snapshots at 00:00 UTC |
| OS | Ubuntu 24.04 LTS, Docker 29 + Compose v2, fail2ban on SSH |
| Access | `ssh -p 2222 -i D:\keys\pd.pem ubuntu@13.60.152.123` (port 2222: the owner's ISP blocks outbound 22) |
| Code | `/home/ubuntu/pd` — a plain copy of the project (no git checkout, no GitHub credentials on the server) |

## Containers (`docker-compose.server.yml`)

| Service | Command | Notes |
|---|---|---|
| `db` | `postgres:17` | tuned for 2 GB (shared_buffers 256 MB); port 5432 is **not** published |
| `screener` | `python screener.py --loop` | `restart: unless-stopped`; the only screener — never run a second one on the laptop (one Coinalyze key, 40 calls/min) |
| `dashboard` | `python dashboard.py --host 0.0.0.0 --no-browser` | published on `127.0.0.1:8766` of the server only; reached through an SSH tunnel |

The image (`Dockerfile`) holds Python 3.14 and the pinned libraries from `requirements.lock.txt` — the same versions
as the laptop's `.venv`, because the joblib model files are tied to the LightGBM / scikit-learn versions. The code is
mounted into the containers (`./:/app`), so a code update needs only a restart, not a rebuild.

Secrets: the server's `~/pd/.env` holds only `COINALYZE_API_KEY` and `POSTGRES_PASSWORD` (mode 600). `DATABASE_URL`
is set by Compose (`db` host) and takes precedence over `.env`. The Binance key is not needed on the server.

## Everyday commands

From the laptop (Git Bash, project folder):

```bash
bash tools/deploy.sh                    # code only
bash tools/deploy.sh --restart          # code + restart screener and dashboard
bash tools/deploy.sh --models --restart # after accepting a new model
```

`deploy.sh` copies every file tracked by git plus new non-ignored files. It never overwrites the files the dashboard
edits on the server (`watchlist.txt`, `scan_config.json`) and never copies `.env`; `--first` (initial deploy only) also
copies them, `models/`, the scanner / wake JSON state and the backtest runs.

Dashboard: double-click `tools/server_dashboard.cmd` — it opens the tunnel and `http://localhost:8766/`; closing the
window closes the tunnel. A local `dashboard.py` on the same port must not be running.

On the server (`cd ~/pd`, `C="docker compose -f docker-compose.server.yml"`):

| Task | Command |
|---|---|
| status | `$C ps`, `docker stats --no-stream`, `free -m` |
| screener log | `$C logs -f --tail 100 screener` |
| restart | `$C restart screener dashboard` |
| rebuild after `requirements.lock.txt` changed | `$C up -d --build` |
| psql | `$C exec db psql -U postgres -d pump_detector` |

## Labeling, training and backtests (laptop)

The server database is the main one; the laptop's database is a copy refreshed on demand.

| Task | Steps |
|---|---|
| Label | `bash tools/pull_db.sh` → `python label.py` → `bash tools/deploy.sh --labels` |
| Train | as above, then **Перенавчити** → compare → **Прийняти** in `label.py` → `bash tools/deploy.sh --models --restart` |
| Backtest | `bash tools/pull_db.sh` → `python backtest.py …` on the laptop → `bash tools/deploy.sh --backtests` to show the run in the dashboard |

`pull_db.sh` replaces the local database, so anything that exists only locally (e.g. a coin loaded in `label.py`
but not yet shipped with `--labels`) is lost; labels are safe in `pumps.csv`. `--labels` also runs `fetch.py --new`
on the server, so a newly added training ticker gets its hourly Coinalyze history there.

## Database migration (done once, 2026-10-06)

`pg_dump -Fc` of the laptop database (4.8 GB on disk → 355 MB dump) → `scp` → `pg_restore --clean -j 2` on the server
(3 minutes; 3.5 GB after restore, row counts checked). The laptop screener was stopped right after; the server screener
filled the gap itself (it always fetches the actual gap since the last stored bar).

## Memory notes

The journal used to load the full raw history of every scored ticker at once — a 1.4 GB peak that froze the dashboard
on the 2 GB host. `journal._series` now loads raw data per ticker from its first score (minus 5 days for ATR warm-up)
and keeps only the aggregated series: 307 MB peak, identical trades (differences ≤ 2e-7 from ATR rounding), and about
twice as fast. Running backtests from the dashboard on the server is still heavy — run them on the laptop.

## Costs

AWS credits ($100) cover the $12 plan until 2027-04-06. After that: switch to the paid plan or move (e.g. Hetzner) —
the move is a database dump plus the same Compose file.
