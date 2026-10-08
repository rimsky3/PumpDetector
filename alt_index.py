"""
alt_index.py — ALT-EW: рівноважений індекс альтів на 5m-барах для ознак «власного руху» 15m-моделі (2026-10-08).

Експеримент experiments/idio_features.py (6 міс., чесний список монет): ознаки руху монети ВІДНОСНО ринку альтів
(бета, власна доходність, Amihud, волатильність) — PF 0.89 → 0.93, кращий у 5 з 6 місяців. «Ринок» там — середня
log-доходність усіх альтів на барі. Тут той самий індекс для робочого конвеєра:

    ret(t) = середнє по монетах log(close_t / close_{t-5хв}),  лише якщо монет з даними ≥ MIN_COINS

Монети — ті самі ~250 ліквідних альтів Binance, що й у market.py (оборот 5 млн – 1 млрд $, без BTC / ETH / стейблів).
5m, а не 15m: ковзна оцінка скринера рахує 15m-бари зі зсувом :05 / :10, і їй потрібна доходність індексу за той самий
бар — features.alt_index_bars складає її з трьох 5m-барів (сума log-доходностей = доходність за 15 хв).

Таблиця alt_index (ts — відкриття 5m-бару). Історія — з архіву data.binance.vision (місячні / денні файли 5m, на диск
не кешуються), хвіст — REST /fapi/v1/klines (короткий запит на монету, вага 1). screener.py --loop оновлює кожні 5 хв
у власному потоці (о :00:08 після закриття бару — до циклу :02 встигає), ensure() — якщо потік ще не встиг.

    python alt_index.py                       # дотягнути (перший раз — історія з HISTORY_FROM, ~10–15 хв)
    python alt_index.py --from 2026-05-01     # історія з іншої дати (перезапише ці бари)
    python alt_index.py --show                # покриття таблиці
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
BAR = pd.Timedelta(minutes=5)
MIN_COINS = 50                 # менше монет на барі — індекс не пишемо (наступне оновлення спробує ще раз)
HISTORY_FROM = "2026-06-01"    # навчальна 15m-історія тікерів — з липня; бета й Amihud беруть 8–9 днів до бару
REST_MAX_BARS = 999            # більший пропуск — спершу з архіву (повні дні), REST — лише хвіст
WORKERS = 6
WEIGHT_PER_SEC = 20            # REST: ~250 запитів ваги 1 за ~13 с (ліміт Binance 2400 / хв, скринер і сканер теж там)
DDL = """CREATE TABLE IF NOT EXISTS alt_index (
    ts TIMESTAMPTZ PRIMARY KEY,          -- відкриття 5m-бару; значення відоме на його закритті (ts + 5 хв)
    ret DOUBLE PRECISION NOT NULL,       -- середня log-доходність альтів за бар (ALT-EW)
    n INT NOT NULL,                      -- скільки монет на барі
    updated_at TIMESTAMPTZ DEFAULT now()
)"""
_lock = threading.Lock()


def ensure_table(conn) -> None:
    conn.execute(DDL)
    conn.commit()


def coins() -> list[str]:
    import market
    return market.universe()


def _accumulate(series_iter, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Закриття монет (Series, індекс — відкриття 5m-бару) → середня log-доходність і кількість монет на кожному барі grid.
    Пропуск у свічках монети = NaN доходності після нього (а не доходність за кілька барів на одному барі)."""
    s = np.zeros(len(grid))
    n = np.zeros(len(grid), dtype=int)
    full = pd.date_range(grid[0] - BAR, grid[-1], freq=BAR)
    for c in series_iter:
        if c is None or c.empty:
            continue
        c = c[~c.index.duplicated(keep="last")].reindex(full)
        lr = np.log(c.where(c > 0)).diff().reindex(grid).to_numpy()
        ok = np.isfinite(lr)
        s[ok] += lr[ok]
        n[ok] += 1
    df = pd.DataFrame({"ret": s / np.maximum(n, 1), "n": n}, index=grid)
    return df[df["n"] >= MIN_COINS]


def _write(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [(ts.to_pydatetime(), float(r), int(k)) for ts, r, k in zip(df.index, df["ret"], df["n"])]
    conn.cursor().executemany("""INSERT INTO alt_index (ts, ret, n) VALUES (%s, %s, %s)
                                 ON CONFLICT (ts) DO UPDATE SET ret = EXCLUDED.ret, n = EXCLUDED.n, updated_at = now()""", rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------- історія з архіву

def _archive_closes(sym: str, start: pd.Timestamp, end_day: pd.Timestamp) -> pd.Series:
    """5m-закриття з архіву за дні [start, end_day]: повні минулі місяці — місячним файлом, решта — подобово."""
    from binance_archive import _csv
    days = pd.date_range(start.normalize(), end_day.normalize(), freq="D")
    this_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    parts = []
    for m in sorted({d.strftime("%Y-%m") for d in days}):
        mdays = [d for d in days if d.strftime("%Y-%m") == m]
        whole = m < this_month and len(mdays) == pd.Timestamp(f"{m}-01").days_in_month
        files = [_csv("klines", sym, m, "5m", monthly=True, cache=False)] if whole else \
                [_csv("klines", sym, d.strftime("%Y-%m-%d"), "5m", cache=False) for d in mdays]
        parts += [f for f in files if len(f)]
    if not parts:
        return pd.Series(dtype=float)
    k = pd.concat(parts)
    return pd.Series(k["close"].astype(float).to_numpy(),
                     index=pd.to_datetime(k["open_time"].astype("int64"), unit="ms", utc=True)).sort_index()


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def history(conn, start, end_day, syms: list[str] | None = None, log=print) -> int:
    """Індекс з архіву за дні [start, end_day] (архів дня виходить наступного дня — end_day не пізніше вчора)."""
    ensure_table(conn)
    syms = syms or coins()
    start, end_day = _utc(start), _utc(end_day)
    grid = pd.date_range(start, end_day.normalize() + pd.Timedelta(days=1) - BAR, freq=BAR)
    t0, done = time.time(), [0]

    def job(sym):
        c = _archive_closes(sym, start - pd.Timedelta(days=1), end_day)
        done[0] += 1
        if done[0] % 50 == 0:
            log(f"  індекс альтів: архів {done[0]}/{len(syms)} монет, {time.time() - t0:.0f} с")
        return c

    with ThreadPoolExecutor(8) as pool:
        df = _accumulate(pool.map(job, syms), grid)
    n = _write(conn, df)
    log(f"  індекс альтів з архіву: {len(syms)} монет, {n} 5m-барів ({start:%Y-%m-%d} – {end_day:%Y-%m-%d}), "
        f"{time.time() - t0:.0f} с")
    return n


# ---------------------------------------------------------------- хвіст з REST

def _rest_closes(sym: str, start: pd.Timestamp, last_open: pd.Timestamp, lim) -> pd.Series:
    """5m-закриття з REST від start до last_open (включно, лише закриті бари)."""
    import exchanges as X
    need = int((last_open - start) / BAR) + 1
    limit = min(1000, max(need, 1))
    lim.take(1 if limit < 100 else 2 if limit < 500 else 5)
    try:
        page = X.get(X.BINANCE + "/fapi/v1/klines", dict(symbol=sym, interval="5m",
                                                         startTime=int(start.timestamp() * 1000), limit=limit))
    except Exception:
        return pd.Series(dtype=float)
    if not page:
        return pd.Series(dtype=float)
    c = pd.Series([float(r[4]) for r in page], index=pd.to_datetime([r[0] for r in page], unit="ms", utc=True))
    return c[c.index <= last_open]


def last_ts(conn):
    try:
        r = conn.execute("SELECT max(ts) FROM alt_index").fetchone()
    except Exception:
        conn.rollback()
        return None
    return pd.Timestamp(r[0]).tz_convert("UTC") if r and r[0] else None


def first_missing(conn, since: pd.Timestamp, until: pd.Timestamp):
    """Перший 5m-бар у [since, until], якого немає в таблиці (дірка: архів дня ще не вийшов, REST-збій), або None."""
    r = conn.execute("""SELECT min(g) FROM generate_series(%s::timestamptz, %s::timestamptz, interval '5 minutes') g
                        WHERE NOT EXISTS (SELECT 1 FROM alt_index a WHERE a.ts = g)""",
                     (since.to_pydatetime(), until.to_pydatetime())).fetchone()
    return pd.Timestamp(r[0]).tz_convert("UTC") if r and r[0] else None


def update(conn, log=print) -> int:
    """Дотягнути індекс до останнього закритого 5m-бару. Порожня таблиця — історія з HISTORY_FROM; пропуск більший за
    REST_MAX_BARS — повні дні з архіву, хвіст з REST. Дірки за останні REST_MAX_BARS барів (напр. день, архів якого ще не
    вийшов під час історії) теж дозаповнюються з REST — інакше лишились би назавжди: дотягуємо лише після останнього бару."""
    import market
    with _lock:
        ensure_table(conn)
        now = pd.Timestamp(datetime.now(timezone.utc))
        last_open = now.floor(BAR) - BAR                       # останній закритий 5m-бар
        last = last_ts(conn)
        first = (last + BAR) if last is not None else pd.Timestamp(HISTORY_FROM, tz="UTC")
        if first > last_open:
            return 0
        syms = coins()
        total = 0
        if (last_open - first) / BAR > REST_MAX_BARS:
            yesterday = now.normalize() - pd.Timedelta(days=1)
            if first < yesterday + pd.Timedelta(days=1):
                total += history(conn, first.normalize(), yesterday, syms, log)
            first = max(first, now.normalize())
        window = last_open - REST_MAX_BARS * BAR
        hole = first_missing(conn, max(window, pd.Timestamp(HISTORY_FROM, tz="UTC")), last_open)
        if hole is None:
            return total
        first = max(min(first, hole), window)                 # старіші дірки — лише python alt_index.py --from
        lim = market._Limiter(WEIGHT_PER_SEC)
        t0 = time.time()
        with ThreadPoolExecutor(WORKERS) as pool:
            got = pool.map(lambda s: _rest_closes(s, first - BAR, last_open, lim), syms)
            df = _accumulate(got, pd.date_range(first, last_open, freq=BAR))
        total += _write(conn, df)
        if (last_open - first) / BAR > 3:
            log(f"  індекс альтів: +{len(df)} 5m-барів з REST ({len(syms)} монет, {time.time() - t0:.0f} с)")
        return total


def ensure(conn, need_open: pd.Timestamp, log=print) -> bool:
    """Є бар need_open (відкриття 5m-бару)? Якщо ні — дотягнути зараз (потік watcher зазвичай уже встиг). Лише короткий
    хвіст з REST: порожню таблицю чи пропуск у дні (історія з архіву, хвилини) тут не заповнюємо — python alt_index.py."""
    last = last_ts(conn)
    if last is not None and last >= need_open:
        return True
    if last is None or (pd.Timestamp(datetime.now(timezone.utc)) - last) / BAR > REST_MAX_BARS:
        log("  ! індекс альтів порожній або з великим пропуском — python alt_index.py (ознаки власного руху будуть NaN)")
        return False
    try:
        update(conn, log)
    except Exception as e:
        log(f"  ! індекс альтів: {type(e).__name__}: {e}")
    last = last_ts(conn)
    return last is not None and last >= need_open


def watcher(db_url: str, delay_s: float = 8.0) -> threading.Thread:
    """Для screener.py --loop: оновлення через delay_s секунд після закриття кожного 5m-бару, у власному потоці."""
    import psycopg

    def run():
        while True:
            try:
                with psycopg.connect(db_url) as conn:
                    update(conn)
            except Exception as e:
                print(f"  ! індекс альтів: {type(e).__name__}: {e}")
            now = pd.Timestamp(datetime.now(timezone.utc))
            time.sleep(max(1.0, ((now.floor(BAR) + BAR + pd.Timedelta(seconds=delay_s)) - now).total_seconds()))
    th = threading.Thread(target=run, daemon=True, name="alt_index")
    th.start()
    return th


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="ALT-EW: 5m-індекс альтів Binance (таблиця alt_index) для ознак власного руху")
    ap.add_argument("--from", dest="start", help="перерахувати історію з архіву з цієї дати (YYYY-MM-DD) до вчора")
    ap.add_argument("--show", action="store_true", help="лише показати покриття таблиці")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        ensure_table(conn)
        if args.start:
            history(conn, args.start, pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1))
        if not args.show:
            update(conn)
        r = conn.execute("SELECT min(ts), max(ts), count(*), round(avg(n)), min(n) FROM alt_index").fetchone()
        print(f"alt_index: {r[0]} – {r[1]}, барів {r[2]}, монет у середньому {r[3]}, мінімум {r[4]}")


if __name__ == "__main__":
    main()
