# Server deployment

Since 2026-10-06 the live part of the system (database, screener, dashboard) runs on a small cloud server 24/7;
labeling, training, backtests and experiments stay on the laptop.

## Host

| | |
|---|---|
| Provider | Vultr, Frankfurt — not a US region: Binance and Bybit block US IPs |
| Plan | Regular Cloud Compute `vc2-2c-4gb`: 2 shared vCPU (no CPU credits), 4 GB RAM (+ 8 GB swap), 80 GB SSD, $20/month; no provider backups (the laptop copy from `pull_db.sh` is the backup) |
| OS | Ubuntu 24.04 LTS, Docker 29 + Compose v2, fail2ban on SSH |
| Access | `ssh -p 2222 -i D:\keys\pd.pem ubuntu@45.76.90.13` (port 2222: the owner's ISP blocks outbound 22; set up at first boot by `tools/vultr_startup.sh`, which also creates the `ubuntu` user) |
| Code | `/home/ubuntu/pd` — a plain copy of the project (no git checkout, no GitHub credentials on the server) |

## Containers (`docker-compose.server.yml`)

| Service | Command | Notes |
|---|---|---|
| `db` | `postgres:17` | tuned for 4 GB (shared_buffers 512 MB); port 5432 is **not** published |
| `screener` | `python screener.py --loop` | `restart: unless-stopped`; the only screener — never run a second one on the laptop (one Coinalyze key, 40 calls/min) |
| `liqrec` | `python liq_recorder.py` | own liquidation feed over WebSocket (docs/liquidations.md); ~45 MB |
| `traderec` | `python trade_recorder.py` | own trade feed (Binance + Bybit) → 5-minute summaries with a footprint `trade_bars`; the screener's b / p marks need it (docs/trades.md) |
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
bash tools/deploy.sh --restart          # code + restart screener and dashboard + start new compose services
bash tools/deploy.sh --models --restart # after accepting a new model
```

`deploy.sh` copies every file tracked by git plus new non-ignored files. It never overwrites the files the dashboard
edits on the server (`watchlist.txt`, `scan_config.json`) and never copies `.env`; `--first` (initial deploy only) also
copies them, `models/`, the scanner / wake JSON state and the backtest runs.

Dashboard from any device: **https://45-76-90-13.sslip.io** — Caddy (`Caddyfile`, service `caddy`) terminates HTTPS with an
automatic Let's Encrypt certificate and asks for a login (basic auth; bcrypt hash in the server `.env` as `PD_PASS_HASH`,
plain password only in `D:\keys\pd_dashboard.txt` on the laptop). `sslip.io` is a free wildcard DNS name that resolves
`45-76-90-13.sslip.io` to `45.76.90.13`; a real domain can replace `PD_HOST` later. Ports 80/443 are open in `ufw`.
Fallback without the password, from the laptop: double-click `tools/server_dashboard.cmd` — it opens the tunnel and `http://localhost:8766/`; closing the
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

## Why not AWS Lightsail (first host, 2026-10-06, a few hours)

The first host was a Lightsail $12 bundle (2 vCPU / 2 GB, $100 of free credits). Lightsail bundles are *burstable*:
the guaranteed baseline is 20% of the CPU (the $24 / 4 GB bundle too); above that the instance spends CPU credits. The
migration and a journal that recomputed non-stop used them up within hours; after that the host was throttled (CPU steal
44–62%), the dashboard list took 36 s, a ticker switch about a minute and the hot coins up to 4 minutes. A non-burstable
Vultr instance at $20 runs the same containers with steal 0. Moving took ~30 minutes with the same Compose file:
stop the screener and recorder on the old host → `pg_dump -Fc` streamed through the laptop → `pg_restore` and start on
the new host (the last step ran server-side under `nohup`, so a power cut at home could not interrupt it).

## Memory notes

The journal used to load the full raw history of every scored ticker at once — a 1.4 GB peak that froze the dashboard
on the 2 GB host. `journal._series` now loads raw data per ticker from its first score (minus 5 days for ATR warm-up)
and keeps only the aggregated series: 307 MB peak, identical trades (differences ≤ 2e-7 from ATR rounding), and about
twice as fast. Running backtests from the dashboard on the server is still heavy — run them on the laptop.

## Speed notes (2026-10-06)

The server's cores are about 3× slower per thread than the laptop, so redundant work that was invisible locally shows up:

- `features.aggregate(..., light=True)` builds only price/OHLC/volume/OI. The ticker list needs just the 24 h changes and
  ATR for the limit hint; the full series (OI candles, funding, liquidations, L/S) is not built for ~64 tickers anymore.
- `features.split_raw` splits the raw frames by ticker with one `groupby` instead of filtering all rows for every ticker.
- `wide()` uses `pivot` instead of `pivot_table` (identical output, checked on 98 tickers × 1h/15m/5m).
- The journal (`J.prepare`) runs in a separate process (`ProcessPoolExecutor`, one worker that loads the models itself),
  so it no longer holds the dashboard process while a chart or the list is requested.

- The ticker list's heavy part (`_watch_list`) is cached: `list_warmer` checks the signals stamp and the wake events
  file every 2 s and recomputes in its own worker process right after new scores; `/api/list` returns the cached
  result (waits for the fresh one only when new scores arrived; otherwise refreshes in the background after 60 s).
  Sources and pending additions are still computed per request.

Result on the server: list ~0.2 s from the cache, ticker chart ~2 s (also while the journal recomputes; before, up to a minute), list 7–8 s (was
13–16 s), journal ~20 s after a new bar, then <1 s. The list output was compared field by field with the old code.

## Costs

AWS credits ($100) cover the $12 plan until 2027-04-06. After that: switch to the paid plan or move (e.g. Hetzner) —
the move is a database dump plus the same Compose file.
