"""
fetch_exchange.py — 15m і 5m дані напряму з Binance і Bybit у ті самі сирі таблиці, що й Coinalyze.

Навіщо: у Coinalyze внутрішньогодинної історії лише 1–3 тижні, а для розмітки пампів з точністю до 5–15 хв
і для 15m-моделі потрібно 60 днів. Біржі віддають те саме (перевірено: значення збігаються з Coinalyze
один в один, experiments/exchange_direct.py), без ключа і з набагато вищими лімітами.

Що пишемо (interval = '15min' / '5min', символ ринку — той самий, що в markets від Coinalyze):
    ohlcv              o, h, l, c, v (монети), bv (лише Binance; у Bybit свічки без buy volume)
    open_interest      USD; свічка зі знімків: open = знімок на початку бару, close — на кінці, high/low — з них
    funding_rate       остання виплачена ставка, %
    long_short_ratio   частка лонг-акаунтів
Ліквідацій біржі не зберігають — їх 15m/5m беремо з Coinalyze (fetch.collect_liquidations, потрібен ключ; --no-liq —
пропустити). OI і long/short Binance — лише за останні 30 днів, Bybit — повні 60.
Годинні дані (1hour) не чіпаємо: їх, як і раніше, збирає fetch.py.

Ринки беруться з таблиці markets, тож спершу тікер має пройти через fetch.py.

    python fetch_exchange.py                # усі тікери з tickers.txt, 15m і 5m, 60 днів
    python fetch_exchange.py QNT AKE --tf 15m
    python fetch_exchange.py --days 3       # дозбір останніх днів
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv

from exchanges import INTERVAL, MS_H, TF, download, normalize, stats
from fetch import TICKERS_FILE, Coinalyze, collect_liquidations, read_tickers_file

EXCHANGES = ["Binance", "Bybit", "OKX"]   # OKX збираємо з 2026-09-30; моделі й графіки поки без нього (features.INTRADAY_EXCHANGES)
WORKERS = 4        # ринки паралельно (запит ~1 с); більше — біржі починають рвати TLS-з'єднання
TABLES = {   # ряд normalize() → таблиця і колонки
    "ohlcv": ("ohlcv", ["o", "h", "l", "c", "v", "bv"]),
    "oi": ("open_interest", ["o", "h", "l", "c"]),
    "funding": ("funding_rate", ["o", "h", "l", "c"]),
    "ls": ("long_short_ratio", ["ratio", "long_pct", "short_pct"]),
}


def upsert(conn, table: str, cols: list[str], symbol: str, interval: str, df) -> int:
    if df.empty:
        return 0
    sql = (f"INSERT INTO {table} (symbol, interval, ts, {', '.join(cols)}) "
           f"VALUES (%s, %s, %s, {', '.join(['%s'] * len(cols))}) "
           f"ON CONFLICT (symbol, interval, ts) DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c in cols))
    vals = df[cols].astype(object).where(df[cols].notna(), None).to_numpy()
    rows = [(symbol, interval, ts.to_pydatetime(), *v) for ts, v in zip(df["ts"], vals)]
    conn.cursor().executemany(sql, rows)
    return len(rows)


def collect(conn, bases: list[str], tfs: list[str], days: float, log=print, exchanges=None) -> int:
    """Завантажити tfs за days днів для ринків Binance/Bybit тікерів bases і записати в базу.
    Спільне для fetch_exchange.py і кнопки «Завантажити» в label.py. Повертає кількість записаних рядів."""
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = int(end - days * 24 * MS_H)
    markets = conn.execute("""
        SELECT symbol, base_asset, exchange_name, symbol_on_exchange FROM markets
        WHERE base_asset = ANY(%s) AND exchange_name = ANY(%s) AND quote_asset = 'USDT'
        ORDER BY base_asset, exchange_name""", (bases, list(exchanges or EXCHANGES))).fetchall()
    missing = sorted(set(bases) - {m[1] for m in markets})
    if missing:
        log(f"Немає ринків {'/'.join(exchanges or EXCHANGES)} у базі: {', '.join(missing)} (спершу python fetch.py, "
            f"або монети немає на цих біржах)")

    def job(m, tf):
        symbol, base, exchange, sym = m
        try:
            return m, tf, normalize(download(exchange, sym, start, end, tf), tf), None
        except Exception as e:  # одна біржа/ринок не має зупиняти решту
            return m, tf, None, e

    done = 0
    # завантаження — паралельно, запис у базу — по черзі в цьому потоці (одне з'єднання)
    with ThreadPoolExecutor(WORKERS) as pool:
        for (symbol, base, exchange, sym), tf, data, err in pool.map(lambda a: job(*a), [(m, tf) for m in markets for tf in tfs]):
            if err is not None:
                log(f"  {base:<8} {exchange:<8} {tf:>3}  помилка: {err}")
                continue
            counts = {}
            for key, (table, cols) in TABLES.items():
                counts[key] = upsert(conn, table, cols, symbol, INTERVAL[tf], data[key])
            conn.commit()
            done += 1
            oi_from = data["oi"]["ts"].min()
            log(f"  {base:<8} {exchange:<8} {tf:>3}  свічок {counts['ohlcv']:>6}, OI {counts['oi']:>6}"
                f" (з {oi_from:%m-%d}), L/S {counts['ls']:>6}" if len(data["oi"]) else
                f"  {base:<8} {exchange:<8} {tf:>3}  свічок {counts['ohlcv']:>6}, OI немає")
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="15m/5m дані з Binance і Bybit у базу (для розмітки і 15m-моделі)")
    ap.add_argument("tickers", nargs="*", help=f"базові активи (без аргументів — з {TICKERS_FILE})")
    ap.add_argument("--tf", nargs="+", choices=["15m", "5m"], default=["15m", "5m"], help="таймфрейми (за замовчуванням обидва)")
    ap.add_argument("--days", type=float, default=60, help="скільки днів історії (за замовчуванням 60)")
    ap.add_argument("--no-liq", action="store_true", help="не брати ліквідації 15m/5m з Coinalyze")
    ap.add_argument("--exchanges", nargs="+", choices=EXCHANGES, default=EXCHANGES, help="з яких бірж (за замовчуванням усі)")
    args = ap.parse_args()

    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("Заповни DATABASE_URL у файлі .env")
    bases = [t.upper() for t in (args.tickers or read_tickers_file(TICKERS_FILE))]
    t0 = time.time()
    with psycopg.connect(db_url) as conn:
        collect(conn, bases, args.tf, args.days, exchanges=args.exchanges)
        api_key = os.getenv("COINALYZE_API_KEY")
        if not args.no_liq and api_key:
            collect_liquidations(Coinalyze(api_key), conn, bases, args.tf, args.days)
        elif not args.no_liq:
            print("Немає COINALYZE_API_KEY — ліквідації пропущено")
    print(f"Готово: {stats['requests']} запитів до бірж, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
