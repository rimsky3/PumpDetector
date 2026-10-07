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
import os
import sys
import tempfile
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
WORKERS = 2               # паралельних днів: великі монети (ETH, PEPE) — мільйони угод на день
CHUNK = 500_000           # угод за раз: пам'ять обмежена незалежно від розміру дня
NB = len(SIZE_BOUNDS) + 1
_session = requests.Session()


def _download(sym: str, day: pd.Timestamp) -> str | None:
    """Архів дня — у тимчасовий файл на диску (не в пам'ять: у великих монет це сотні МБ). None — файлу немає."""
    with _session.get(URL.format(s=quote(sym), d=f"{day:%Y-%m-%d}"), timeout=180, stream=True) as r:
        if r.status_code == 404:
            return None
        r.raise_for_status()
        f = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        with f:
            for part in r.iter_content(1 << 20):
                f.write(part)
    return f.name


def archive_rows(sym: str, day: pd.Timestamp) -> list[tuple] | None:
    """Один день угод Binance → рядки trade_bars (як у записувача). None — файлу немає (ще не викладено / не торгувався).
    CSV читається частинами по CHUNK угод: проміжні суми по барах / рівнях складаються, тож пам'ять не залежить від
    розміру дня (07.10: ETH / PEPE цілком у пам'яті — 1.4 ГБ і OOM у контейнері скринера)."""
    path = _download(sym, day)
    if path is None:
        return None
    sides, ks, fps = [], [], []
    try:
        with zipfile.ZipFile(path) as z:
            name = z.namelist()[0]
            with z.open(name) as fh:
                head = fh.read(20).decode("utf-8", "ignore")
            with z.open(name) as fh:
                reader = pd.read_csv(fh, header=0 if head.startswith("agg_trade_id") else None, chunksize=CHUNK,
                                     names=["id", "price", "qty", "first", "last", "time", "ibm"], usecols=["price", "qty", "time", "ibm"])
                for t in reader:
                    ibm = t["ibm"] if t["ibm"].dtype == bool else t["ibm"].astype(str).str.lower().eq("true")
                    p = t["price"].to_numpy(float)
                    d = pd.DataFrame({"bar": t["time"].to_numpy() // 300_000 * 300, "buy": ~ibm.to_numpy(),
                                      "usd": p * t["qty"].to_numpy(float), "lvl": np.floor(np.log(p) / FP.LVL).astype(np.int64)})
                    d["k"] = np.searchsorted(SIZE_BOUNDS, d["usd"], side="right")      # = bisect_right у записувача
                    sides.append(d.groupby(["bar", "buy"])["usd"].agg(["count", "sum", "max"]))
                    ks.append(d.groupby(["bar", "buy", "k"])["usd"].agg(["count", "sum"]))
                    fps.append(d.groupby(["bar", "lvl", "buy"])["usd"].sum())
    finally:
        os.remove(path)
    if not sides:
        return []
    sd = pd.concat(sides).groupby(level=[0, 1]).agg({"count": "sum", "sum": "sum", "max": "max"})
    kd = pd.concat(ks).groupby(level=[0, 1, 2]).sum()
    fd = pd.concat(fps).groupby(level=[0, 1, 2]).sum()
    rows = []
    for bar in sd.index.get_level_values(0).unique():
        side = {buy: sd.loc[(bar, buy)] if (bar, buy) in sd.index else None for buy in (True, False)}
        bn, sn, bu, su = (np.zeros(NB, int), np.zeros(NB, int), np.zeros(NB), np.zeros(NB))
        for (buy, k), r in kd.loc[bar].iterrows():
            if buy:
                bn[k], bu[k] = r["count"], r["sum"]
            else:
                sn[k], su[k] = r["count"], r["sum"]
        f = fd.loc[bar]
        lvls = f.index.get_level_values(0)
        lo, hi = int(lvls.min()), int(lvls.max())
        fb, fs = np.zeros(hi - lo + 1), np.zeros(hi - lo + 1)
        for (lv, buy), v in f.items():
            (fb if buy else fs)[lv - lo] += v
        nb = int(side[True]["count"]) if side[True] is not None else 0
        ns = int(side[False]["count"]) if side[False] is not None else 0
        rows.append((datetime.fromtimestamp(int(bar), tz=timezone.utc), nb, ns,
                     float(side[True]["sum"]) if nb else 0.0, float(side[False]["sum"]) if ns else 0.0,
                     float(side[True]["max"]) if nb else None, float(side[False]["max"]) if ns else None,
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
    with_fp = sorted({b for b in bases if FP._binance_symbol(conn, b)})
    if jobs:
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
    if m15 is not None and days:
        # мітки — на всьому періоді для тікерів з футпрінтом (не лише на щойно дозаповнених днях: перерваний прогін
        # лишив би дні без міток); уже записані мітки не змінюються (ON CONFLICT DO NOTHING)
        for base in with_fp:
            try:
                found = FP.detect_range(conn, base, days[0], days[-1] + pd.Timedelta(days=1) - FP.BAR, m15)
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
