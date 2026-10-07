"""
binance_archive.py — історія Binance з архіву data.binance.vision у ті самі сирі таблиці, що й fetch_exchange (2026-10-07).

API Binance віддає OI і частку лонгів лише за 30 днів, тож у базі OI Binance починався на місяць пізніше за свічки:
навчальна історія 15m-моделі в серпні мала OI і long/short лише з Bybit. Архів — щоденні файли з грудня 2021:
    metrics          OI (монети) і частка лонг-акаунтів (count_long_short_ratio) кожні 5 хв
    klines/<tf>      свічки з обсягом покупок (taker buy)
Funding — з API (історія без обмежень). Усе йде через exchanges.normalize і fetch_exchange.upsert — ті самі правила
вирівнювання часу, що й для API; --check звіряє архів з тим, що в базі (з API) на спільних днях.

    python binance_archive.py NMR ZEC --from 2026-07-25 --to 2026-10-05          # дозаповнити 15m
    python binance_archive.py NMR --from 2026-09-10 --to 2026-09-20 --check      # звірка з базою, нічого не пише
"""

import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from exchanges import INTERVAL, TF, binance_funding, normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data", "binance_archive")          # сирі денні файли (метрики ~10 КБ, свічки 15m ~5 КБ)
BASE_URL = "https://data.binance.vision/data/futures/um/daily"
WORKERS = 8
_S = requests.Session()
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _csv(kind: str, symbol: str, day: str, tf: str | None = None, monthly: bool = False) -> pd.DataFrame:
    """Денний (або місячний: day = YYYY-MM, monthly=True — лише свічки) файл архіву; кеш на диску; немає файлу — порожній
    кадр, не кешується (архів за день з'являється наступного дня, за місяць — на початку наступного)."""
    sub = f"klines/{symbol}/{tf}/{symbol}-{tf}-{day}" if kind == "klines" else f"metrics/{symbol}/{symbol}-metrics-{day}"
    path = os.path.join(CACHE, ("monthly/" if monthly else "") + sub + ".parquet")
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:               # обрізаний файл (вимкнення світла під час запису) — видалити й скачати заново
            os.remove(path)
    url = f"{BASE_URL.replace('/daily', '/monthly')}/{sub}.zip" if monthly else f"{BASE_URL}/{sub}.zip"
    for attempt in range(4):
        try:
            r = _S.get(url, timeout=120)
            if r.status_code == 404:
                return pd.DataFrame()
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.open(z.namelist()[0]).read().decode()
            break
        except (requests.RequestException, zipfile.BadZipFile):
            time.sleep(3 + 3 * attempt)
    else:
        return pd.DataFrame()
    if kind == "klines":                               # старі файли без заголовка, нові — із заголовком
        has_header = raw.split(",", 1)[0].strip() == "open_time"
        df = pd.read_csv(io.StringIO(raw), header=0 if has_header else None, names=None if has_header else KLINE_COLS)
    else:
        df = pd.read_csv(io.StringIO(raw))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path + ".tmp")          # атомарно: обрив посеред запису не лишає битого файлу
    os.replace(path + ".tmp", path)
    return df


def _days(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def download(symbol: str, start: str, end: str, tf: str = "15m") -> dict[str, pd.DataFrame]:
    """Дані у форматі exchanges.download (для normalize): klines, oi (монети), ls (частка лонг-акаунтів), funding."""
    days = _days(start, end)
    # свічки: повні завершені місяці — одним місячним файлом (у ~30 разів менше запитів), решта — подобово
    this_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    months = {d[:7] for d in days}
    full = {m for m in months if m < this_month and all(x in days for x in _days(f"{m}-01", (pd.Timestamp(f"{m}-01") + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")))}
    with ThreadPoolExecutor(WORKERS) as pool:
        kl = list(pool.map(lambda m: _csv("klines", symbol, m, tf, monthly=True), sorted(full)))
        kl += list(pool.map(lambda d: _csv("klines", symbol, d, tf), [d for d in days if d[:7] not in full]))
        mt = list(pool.map(lambda d: _csv("metrics", symbol, d), days))
    kl = pd.concat([k for k in kl if len(k)]) if any(len(k) for k in kl) else pd.DataFrame(columns=KLINE_COLS)
    mt = pd.concat([m for m in mt if len(m)]) if any(len(m) for m in mt) else pd.DataFrame(columns=["create_time"])
    k = pd.DataFrame({"ts": pd.to_datetime(kl["open_time"].astype("int64"), unit="ms", utc=True),
                      "o": kl["open"].astype(float), "h": kl["high"].astype(float), "l": kl["low"].astype(float),
                      "c": kl["close"].astype(float), "v": kl["volume"].astype(float),
                      "bv": kl["taker_buy_volume"].astype(float)}).drop_duplicates("ts").sort_values("ts")
    ms = TF[tf]["ms"]
    if len(mt):
        # create_time архіву на 5 хв РАНІШЕ за мітку API того самого знімка (звірено 07.10: зі зсувом +5 хв OI збігається
        # з API в 100% барів, частка лонгів — до округлення; без зсуву — розбіжність ~0.03–2%)
        t = pd.to_datetime(mt["create_time"], utc=True) + pd.Timedelta(minutes=5)
        # як API з period=<tf>: лише знімки на межах барів (через Timedelta — одиниця часу в pandas буває нс або мкс)
        on_grid = ((t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)) % ms == 0
        mt, t = mt[on_grid.to_numpy()], t[on_grid.to_numpy()]
        oi = pd.DataFrame({"ts": t.to_numpy(), "value": mt["sum_open_interest"].astype(float).to_numpy()})
        r = mt["count_long_short_ratio"].astype(float).to_numpy()
        ls = pd.DataFrame({"ts": t.to_numpy(), "value": r / (1 + r)})    # longAccount = ratio / (1 + ratio)
        # архів зрідка має нулі (збій запису: OI = 0 на 1–2 знімках) — це пропуск, а не значення
        oi, ls = oi[oi["value"] > 0], ls[(ls["value"] > 0) & (ls["value"] < 1)]
    else:
        oi = ls = pd.DataFrame({"ts": pd.to_datetime([], utc=True), "value": []})
    s_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    e_ms = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    funding = binance_funding(symbol, s_ms - 3 * 86_400_000, e_ms)
    return {"klines": k, "oi": oi.drop_duplicates("ts").sort_values("ts"), "ls": ls.drop_duplicates("ts").sort_values("ts"),
            "funding": funding}


def markets(conn, bases: list[str]) -> list[tuple[str, str, str]]:
    return conn.execute("""SELECT symbol, base_asset, symbol_on_exchange FROM markets WHERE base_asset = ANY(%s)
                           AND exchange_name = 'Binance' AND quote_asset = 'USDT' ORDER BY base_asset""", (bases,)).fetchall()


def collect(conn, bases: list[str], start: str, end: str, tfs=("15m",), log=print) -> int:
    """Архів → normalize → ті самі таблиці (upsert). Повертає кількість записаних рядів."""
    import fetch_exchange
    total = 0
    for symbol, base, sym in markets(conn, bases):
        for tf in tfs:
            data = normalize(download(sym, start, end, tf), tf)
            n = {key: fetch_exchange.upsert(conn, table, cols, symbol, INTERVAL[tf], data[key])
                 for key, (table, cols) in fetch_exchange.TABLES.items()}
            conn.commit()
            total += sum(n.values())
            log(f"  {base:<8} {tf:>3}  свічок {n['ohlcv']:>6}, OI {n['oi']:>6}, L/S {n['ls']:>6}")
    return total


def check(conn, base: str, start: str, end: str, tf: str = "15m") -> None:
    """Звірка архіву з базою (рядки з API) на спільних барах: відносна різниця OI, частки лонгів, ціни, обсягу покупок."""
    rows = markets(conn, [base])
    if not rows:
        sys.exit(f"{base}: немає ринку Binance в markets")
    symbol, _, sym = rows[0]
    arch = normalize(download(sym, start, end, tf), tf)
    q = lambda t, cols: pd.DataFrame(conn.execute(f"SELECT ts, {', '.join(cols)} FROM {t} WHERE symbol = %s AND interval = %s "   # noqa: E731
                                                  f"AND ts >= %s AND ts < %s ORDER BY ts", (symbol, INTERVAL[tf], start,
                                                  pd.Timestamp(end) + pd.Timedelta(days=1))).fetchall(), columns=["ts", *cols])
    for name, (table, a_col, d_col) in {"OI, USD (close)": ("open_interest", "c", "c"), "частка лонгів": ("long_short_ratio", "long_pct", "long_pct"),
                                        "ціна": ("ohlcv", "c", "c"), "обсяг покупок": ("ohlcv", "bv", "bv")}.items():
        key = {"open_interest": "oi", "long_short_ratio": "ls", "ohlcv": "ohlcv"}[table]
        db = q(table, [d_col]).set_index("ts")[d_col].astype(float)
        db.index = pd.to_datetime(db.index, utc=True)
        a = arch[key].set_index("ts")[a_col].astype(float)
        j = pd.concat([a.rename("arch"), db.rename("db")], axis=1, sort=True).dropna()
        if j.empty:
            print(f"  {name:<16} спільних барів немає (архів {len(a)}, база {len(db)})")
            continue
        rel = ((j["arch"] - j["db"]).abs() / j["db"].abs().where(j["db"] != 0)).fillna(0)
        print(f"  {name:<16} спільних барів {len(j):>5}; відносна різниця медіана {rel.median():.2e}, макс {rel.max():.2e}, "
              f"збіг до 1e-6 у {(rel < 1e-6).mean():.1%}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Історія Binance з архіву data.binance.vision (OI, частка лонгів, свічки) у базу")
    ap.add_argument("tickers", nargs="+", help="базові активи, напр. NMR ZEC")
    ap.add_argument("--from", dest="start", required=True, help="перший день, YYYY-MM-DD")
    ap.add_argument("--to", dest="end", required=True, help="останній день, YYYY-MM-DD (архів за день виходить наступного дня)")
    ap.add_argument("--tf", nargs="+", choices=["15m", "5m"], default=["15m"], help="таймфрейми (за замовчуванням 15m)")
    ap.add_argument("--check", action="store_true", help="лише звірити архів з базою, нічого не писати")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        bases = [t.upper() for t in args.tickers]
        if args.check:
            for b in bases:
                print(b)
                check(conn, b, args.start, args.end, args.tf[0])
        else:
            collect(conn, bases, args.start, args.end, args.tf)


if __name__ == "__main__":
    main()
