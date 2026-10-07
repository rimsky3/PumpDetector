"""
trade_backfill.py — історія футпрінту з архіву угод Binance → trade_bars + історичні мітки b / p (2026-10-07).

Записувач угод (trade_recorder.py) бачить лише те, що сталося після його запуску. Архів Binance
(data.binance.vision/data/futures/um/daily/aggTrades) — ті самі угоди (aggTrade), що й живий потік, за будь-який день,
викладається наступного дня. Скрипт «програє» архів у ту саму таблицю trade_bars і в тому самому форматі (5-хв бари:
агресор, кошики розміру, найбільша угода, футпрінт на рівнях 0.1%) — тож графік футпрінту й пошук міток працюють на
історії так само, як наживо. Потім на дозаповнених днях шукає мітки b / p у контексті моделі — з позначкою live = false
(історія: бліді на графіку, не в сповіщення, окремий рядок статистики).

Лише Binance (мітки й графік футпрінту — з Binance). Дні, де вже є ≥ FULL_DAY 5-хв барів (записувач або минулий прогін),
пропускаються; рядки записувача не перезаписуються (ON CONFLICT DO NOTHING) — повторний запуск дешевий.

Скринер запускає його окремим процесом (з nice — сигнали не гальмує): при старті (усі тікери списку за DAYS днів),
раз на добу після публікації архіву (за вчора — закриває дірки, коли записувач не працював) і для нових тікерів списку.

    python trade_backfill.py                    # тікери списку скринера (watchlist + авто), 30 днів
    python trade_backfill.py ORCA CAP --days 7  # вибрані тікери
    python trade_backfill.py --yesterday        # лише вчорашній день (щоденне дозаповнення)
"""

import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd
import psycopg
import requests
from dotenv import load_dotenv

import footprint as FP
from trade_recorder import DDL as TRADE_DDL, SIZE_BOUNDS

ROOT = os.path.dirname(os.path.abspath(__file__))
URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
DAYS = 30
FULL_DAY = 280            # з 288 5-хв барів: день вважаємо вже заповненим
WORKERS = 4
NB = len(SIZE_BOUNDS) + 1
_session = requests.Session()


def archive_rows(sym: str, day: pd.Timestamp) -> list[tuple] | None:
    """Один день угод Binance → рядки trade_bars (як у записувача). None — файлу немає (ще не викладено / не торгувався)."""
    r = _session.get(URL.format(s=quote(sym), d=f"{day:%Y-%m-%d}"), timeout=180)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0])
    head = raw[:20].decode("utf-8", "ignore")
    t = pd.read_csv(io.BytesIO(raw), header=0 if head.startswith("agg_trade_id") else None,
                    names=["id", "price", "qty", "first", "last", "time", "ibm"], usecols=["price", "qty", "time", "ibm"])
    if t.empty:
        return []
    ibm = t["ibm"] if t["ibm"].dtype == bool else t["ibm"].astype(str).str.lower().eq("true")
    p = t["price"].to_numpy(float)
    d = pd.DataFrame({"bar": t["time"].to_numpy() // 300_000 * 300, "buy": ~ibm.to_numpy(), "usd": p * t["qty"].to_numpy(float),
                      "lvl": np.floor(np.log(p) / FP.LVL).astype(np.int64)})
    d["k"] = np.searchsorted(SIZE_BOUNDS, d["usd"], side="right")      # = bisect_right у записувача
    rows = []
    for bar, g in d.groupby("bar"):
        b, s = g[g["buy"]], g[~g["buy"]]
        bn, sn = np.bincount(b["k"], minlength=NB), np.bincount(s["k"], minlength=NB)
        bu, su = np.bincount(b["k"], weights=b["usd"], minlength=NB), np.bincount(s["k"], weights=s["usd"], minlength=NB)
        lo, hi = int(g["lvl"].min()), int(g["lvl"].max())
        fb = np.bincount(b["lvl"] - lo, weights=b["usd"], minlength=hi - lo + 1)
        fs = np.bincount(s["lvl"] - lo, weights=s["usd"], minlength=hi - lo + 1)
        rows.append((datetime.fromtimestamp(int(bar), tz=timezone.utc), len(b), len(s), float(b["usd"].sum()), float(s["usd"].sum()),
                     float(b["usd"].max()) if len(b) else None, float(s["usd"].max()) if len(s) else None,
                     bn.tolist(), sn.tolist(), bu.tolist(), su.tolist(), lo, fb.tolist(), fs.tolist()))
    return rows


def missing_days(conn, sym: str, days: list[pd.Timestamp]) -> list[pd.Timestamp]:
    have = dict(conn.execute("""SELECT date_trunc('day', ts), count(*) FROM trade_bars
                                WHERE exchange = 'Binance' AND symbol = %s AND ts >= %s AND fp_lvl0 IS NOT NULL GROUP BY 1""",
                             (sym, days[0].to_pydatetime())).fetchall()) if days else {}
    have = {pd.Timestamp(k).tz_convert("UTC").normalize(): v for k, v in have.items()}
    return [d for d in days if have.get(d, 0) < FULL_DAY]


def backfill(conn, bases: list[str], days: list[pd.Timestamp], m15: dict | None, log=print) -> dict:
    """Дозаповнити trade_bars з архіву для bases за days (дні UTC) і знайти історичні мітки b / p."""
    conn.execute(TRADE_DDL)
    FP.ensure_table(conn)
    stat = {"days": 0, "rows": 0, "marks": 0}
    jobs = []
    for base in bases:
        sym = FP._binance_symbol(conn, base)
        if sym is None:
            continue
        for d in missing_days(conn, sym[1], days):
            jobs.append((base, sym[1], d))
    if not jobs:
        return stat
    log(f"архів угод Binance: {len(jobs)} днів для {len({j[0] for j in jobs})} тікерів")
    filled: dict[str, list] = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(archive_rows, sym, d): (base, sym, d) for base, sym, d in jobs}
        for fu in futs:
            base, sym, d = futs[fu]
            try:
                rows = fu.result()
            except Exception as e:
                log(f"  ! {base} {d:%Y-%m-%d}: {e}")
                continue
            if not rows:
                continue
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO trade_bars (exchange, symbol, ts, n_buy, n_sell, v_buy, v_sell, max_buy, max_sell,
                                            buy_n, sell_n, buy_usd, sell_usd, fp_lvl0, fp_buy, fp_sell)
                    VALUES ('Binance', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                                [(sym,) + r for r in rows])
            conn.commit()
            stat["days"] += 1
            stat["rows"] += len(rows)
            filled.setdefault(base, []).append(d)
    if m15 is not None:
        for base, ds in filled.items():
            try:
                found = FP.detect_range(conn, base, min(ds), max(ds) + pd.Timedelta(days=1) - FP.BAR, m15)
                stat["marks"] += len(found)
            except Exception as e:
                conn.rollback()
                log(f"  ! мітки {base}: {e}")
    log(f"архів угод: {stat['days']} днів, {stat['rows']} 5-хв барів, історичних міток b / p: {stat['marks']}")
    return stat


def day_list(days: int, yesterday: bool) -> list[pd.Timestamp]:
    today = pd.Timestamp.now(tz="UTC").normalize()
    if yesterday:
        return [today - pd.Timedelta(days=1)]
    return list(pd.date_range(today - pd.Timedelta(days=days), today - pd.Timedelta(days=1), freq="D"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Історія футпрінту з архіву угод Binance → trade_bars і історичні мітки b / p")
    ap.add_argument("bases", nargs="*", help="тікери (за замовчуванням — список скринера: watchlist + авто)")
    ap.add_argument("--days", type=int, default=DAYS, help=f"скільки останніх повних днів (за замовчуванням {DAYS})")
    ap.add_argument("--yesterday", action="store_true", help="лише вчорашній день (щоденне дозаповнення)")
    ap.add_argument("--no-marks", action="store_true", help="лише футпрінт, без пошуку історичних міток")
    sys.stdout.reconfigure(encoding="utf-8")         # до parse_args: --help з кирилицею й «→» у консолі Windows
    args = ap.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("Заповни DATABASE_URL у .env")
    if args.bases:
        bases = [b.upper() for b in args.bases]
    else:
        from screener import cycle_bases
        bases = cycle_bases()
    m15 = None
    if not args.no_marks:
        from screener import load_model_15m
        m15 = load_model_15m()
    t0 = time.time()
    with psycopg.connect(db_url) as conn:
        backfill(conn, bases, day_list(args.days, args.yesterday), m15, log=lambda m: print(m, flush=True))
    print(f"готово за {(time.time() - t0) / 60:.1f} хв", flush=True)


if __name__ == "__main__":
    main()
