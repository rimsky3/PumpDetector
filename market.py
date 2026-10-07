"""
market.py — стан ринку альтів (risk on / risk off) для дашборда, журналу і бектесту (2026-10-07).

Експеримент experiments/market_regime.py (6 міс. + вересень): коли > 60% альтів дорожчі, ніж тиждень тому («гарячий
ринок»), Alert моделі гірший — PF 0.78 проти 1.09 (6 міс.), 1.00 проти 1.69 (вересень), у 6 з 6 місяців. BTC сам по собі —
шум. Пояснення: у широкому ралі Alert — рух разом з ринком, він згасає з ринком; коли ринок слабкий, у пампу монети
своя причина.

Тут ті самі ознаки наживо. Скринер щогодини (HH:01) тягне 1h-свічки ліквідних альтів Binance (середній оборот за 30 днів
5 млн $ … 1 млрд $, без стейблів; один короткий запит на монету, ~300 запитів, вага 1) → закриття в кеші
reports/market_closes.pkl → ознаки на закритій годині → таблиця market_breadth. Перший запуск — історія за HISTORY_DAYS.
Дашборд — кнопка «Market» над графіком (/api/market), журнал і бектест — режим «ринок не гарячий» (allow_fn).

    python market.py            # оновити (перший раз — історія, ~2–3 хв)
    python market.py --show     # останній стан
"""

import argparse
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CLOSES_FILE = os.path.join(ROOT, "reports", "market_closes.pkl")
QVOL_FILE = os.path.join(ROOT, "reports", "market_qvol.pkl")   # $-оборот тих самих годин (Amihud ILLIQ у movers.py)
H = pd.Timedelta(hours=1)
HISTORY_DAYS = 60
MIN_TURNOVER, MAX_TURNOVER = 5e6, 1e9      # «альти»: досить ліквідні, але без BTC / ETH / SOL / XRP / ZEC-гігантів
NOT_ALTS = {"BTCUSDT", "ETHUSDT", "USDCUSDT", "FDUSDUSDT", "BTCDOMUSDT", "PAXGUSDT", "XAUTUSDT", "FRAXUSDT", "USTCUSDT"}
BTC = "BTCUSDT"
MIN_COINS = 50                             # менше монет з історією — ширину не рахуємо
HOT, COLD = 0.60, 0.40                     # частка альтів у плюсі за 7 днів: > HOT — «гарячий», < COLD — «холодний»
STALE = pd.Timedelta(hours=3)              # фільтр режиму: старіший стан — «невідомо», сигнал береться
WORKERS = 4
WEIGHT_PER_SEC = 15                        # ~900 ваги / хв: скринер і сканер теж ходять у Binance (ліміт 2400)
# результат бектесту (experiments/market_regime.py, 07.10) — для підказки в дашборді
BACKTEST = {"hot_pf": 0.78, "rest_pf": 1.09, "sept_hot_pf": 1.00, "sept_rest_pf": 1.69, "months": "6 з 6"}

COLS = ["n", "breadth_24h", "breadth_7d", "alt_med_24h", "alt_med_7d", "alt_vs_btc_24h",
        "btc_ret_24h", "btc_vs_ma7d", "btc_dd_7d", "btc_vol_x"]
DDL = """CREATE TABLE IF NOT EXISTS market_breadth (
    ts TIMESTAMPTZ PRIMARY KEY,          -- відкриття години; значення відомі на її закритті (ts + 1 год)
    n INT,                               -- скільки альтів з історією за 7 днів
    breadth_24h DOUBLE PRECISION, breadth_7d DOUBLE PRECISION,       -- частка альтів у плюсі за 24 год / 7 днів
    alt_med_24h DOUBLE PRECISION, alt_med_7d DOUBLE PRECISION,       -- медіанна зміна альтів
    alt_vs_btc_24h DOUBLE PRECISION,     -- медіана альтів мінус BTC за 24 год
    btc_ret_24h DOUBLE PRECISION, btc_vs_ma7d DOUBLE PRECISION, btc_dd_7d DOUBLE PRECISION, btc_vol_x DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT now()
)"""


def ensure_table(conn) -> None:
    conn.execute(DDL)
    conn.commit()


# ---------------------------------------------------------------- ознаки (спільні з experiments/market_regime.py)

def features(closes: pd.DataFrame) -> pd.DataFrame:
    """Закриття годин (індекс — відкриття години, колонки — символи, BTCUSDT окремо) → ознаки ринку на кожній годині.
    Значення години відомі на її закритті. Без погляду вперед: лише shift / rolling назад."""
    idx = pd.date_range(closes.index.min(), closes.index.max(), freq="1h")
    closes = closes.reindex(idx)
    c = closes[BTC].ffill() if BTC in closes else pd.Series(np.nan, index=idx)
    alts = closes.drop(columns=[BTC], errors="ignore")
    lr = np.log(c).diff()
    vol24 = lr.rolling(24).std()
    f = pd.DataFrame({
        "btc_ret_24h": c / c.shift(24) - 1,
        "btc_vs_ma7d": c / c.rolling(168).mean() - 1,
        "btc_dd_7d": c / c.rolling(168).max() - 1,
        "btc_vol_x": vol24 / vol24.rolling(720, min_periods=240).median(),
    }, index=idx)
    r24, r7 = alts / alts.shift(24) - 1, alts / alts.shift(168) - 1
    n24, n7 = r24.notna().sum(axis=1), r7.notna().sum(axis=1)
    f["n"] = n7
    f["breadth_24h"] = ((r24 > 0).sum(axis=1) / n24).where(n24 >= MIN_COINS)
    f["breadth_7d"] = ((r7 > 0).sum(axis=1) / n7).where(n7 >= MIN_COINS)
    f["alt_med_24h"] = r24.median(axis=1).where(n24 >= MIN_COINS)
    f["alt_med_7d"] = r7.median(axis=1).where(n7 >= MIN_COINS)
    f["alt_vs_btc_24h"] = f["alt_med_24h"] - f["btc_ret_24h"]
    return f[COLS]


def state_of(b7: float | None) -> str:
    if b7 is None or not np.isfinite(b7):
        return "unknown"
    return "hot" if b7 > HOT else "cold" if b7 < COLD else "neutral"


# ---------------------------------------------------------------- дані з Binance

class _Limiter:
    """Вага запитів Binance: не більше WEIGHT_PER_SEC у середньому (спільна для потоків)."""
    def __init__(self, rate: float):
        self.rate, self.lock, self.next = rate, threading.Lock(), time.monotonic()

    def take(self, w: int) -> None:
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now) + w / self.rate
            wait = self.next - w / self.rate - now
        if wait > 0:
            time.sleep(wait)


def universe() -> list[str]:
    """Ліквідні альти: середній денний оборот за 30 днів з кешу сканера (reports/scan_turnover_avg.json)."""
    import scanner
    vals = (scanner._read_json(scanner.AVG_FILE, {}) or {}).get("values") or {}
    if not vals:                                   # кешу немає (ноутбук без сканера) — сканер порахує сам (~75 с)
        vals = {m["symbol"]: scanner.liquidity(m) for m in scanner.universe()}
    return sorted(s for s, v in vals.items() if v and MIN_TURNOVER <= v <= MAX_TURNOVER and s.endswith("USDT")
                  and s not in NOT_ALTS)


def _klines(sym: str, start: pd.Timestamp, now: pd.Timestamp, lim: _Limiter) -> pd.DataFrame:
    """1h-свічки з start (включно) до останньої закритої години: c — закриття, q — оборот у $ (quote volume)."""
    import exchanges as X
    out, t = [], int(start.timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    while t < end:
        need = int((end - t) / 3_600_000) + 1
        limit = min(1000, max(need, 1))
        lim.take(1 if limit < 100 else 2 if limit < 500 else 5)
        try:
            page = X.get(X.BINANCE + "/fapi/v1/klines", dict(symbol=sym, interval="1h", startTime=t, limit=limit))
        except Exception:
            break                                   # символ зник / помилка — без нього
        if not page:
            break
        out += [(r[0], float(r[4]), float(r[7])) for r in page]
        if len(page) < limit:
            break
        t = page[-1][0] + 3_600_000
    if not out:
        return pd.DataFrame(columns=["c", "q"], dtype=float)
    df = pd.DataFrame(out, columns=["ms", "c", "q"])
    df.index = pd.to_datetime(df.pop("ms"), unit="ms", utc=True)
    df = df[~df.index.duplicated(keep="last")]
    return df[df.index + H <= now]                  # лише закриті години


def _load(path: str) -> pd.DataFrame:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return pd.DataFrame()


def load_closes() -> pd.DataFrame:
    return _load(CLOSES_FILE)


def load_qvol() -> pd.DataFrame:
    return _load(QVOL_FILE)


def _save(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(df, f)
    os.replace(path + ".tmp", path)


def update(conn, log=print) -> int:
    """Дотягнути закриття годин, перерахувати ознаки за останні 2 дні (або всю історію вперше) → market_breadth.
    Повертає кількість записаних годин."""
    ensure_table(conn)
    now = pd.Timestamp(datetime.now(timezone.utc))
    closes, qvol = load_closes(), load_qvol()
    syms = universe() + [BTC]
    first = closes.empty
    hist_start = (now - pd.Timedelta(days=HISTORY_DAYS)).floor("h")
    lim = _Limiter(WEIGHT_PER_SEC)

    def nxt(df, sym):
        have = df[sym].dropna() if sym in df else pd.Series(dtype=float)
        return have.index.max() + H if len(have) else hist_start

    def job(sym):                                   # оборот з'явився пізніше за закриття — тоді дотягуємо з його початку
        return sym, _klines(sym, min(nxt(closes, sym), nxt(qvol, sym)), now, lim)

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        got = dict(pool.map(job, syms))
    keep = set(syms) | {BTC}                        # монети, що випали з універсуму, — більше не рахуємо

    def merge(old, col):
        new = pd.DataFrame({s: v[col] for s, v in got.items() if len(v)})
        df = new if old.empty else pd.concat([old, new]).groupby(level=0).last()
        df = df[df.index >= hist_start].sort_index()
        return df[[c for c in df.columns if c in keep]]
    closes, qvol = merge(closes, "c"), merge(qvol, "q")
    _save(closes, CLOSES_FILE)
    _save(qvol, QVOL_FILE)
    try:                                            # кнопка «Movers» у дашборді — з тих самих годин
        import movers
        movers.save(closes, qvol)
    except Exception as e:
        log(f"  ! movers: {type(e).__name__}: {e}")

    f = features(closes)
    if not first:
        f = f[f.index >= now - pd.Timedelta(days=2)]
    f = f.dropna(subset=["breadth_7d", "btc_ret_24h"], how="all")
    rows = [(ts.to_pydatetime(), *[None if pd.isna(v) else (int(v) if c == "n" else float(v)) for c, v in zip(COLS, r)])
            for ts, r in zip(f.index, f[COLS].to_numpy())]
    sql = (f"INSERT INTO market_breadth (ts, {', '.join(COLS)}) VALUES (%s, {', '.join(['%s'] * len(COLS))}) "
           f"ON CONFLICT (ts) DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c in COLS) + ", updated_at = now()")
    conn.cursor().executemany(sql, rows)
    conn.commit()
    _cache.clear()
    log(f"  ринок: {len(syms) - 1} альтів, {len(rows)} годин записано за {time.time() - t0:.0f} с"
        + (f" (історія {HISTORY_DAYS} днів)" if first else ""))
    return len(rows)


def watcher(db_url: str, minute: int = 1) -> threading.Thread:
    """Для screener.py --loop: оновлення щогодини о HH:01 (і одразу при старті) у власному потоці — годинна частина
    скринера (Coinalyze) його не затримує."""
    import psycopg

    def run():
        while True:
            try:
                with psycopg.connect(db_url) as conn:
                    update(conn)
            except Exception as e:
                print(f"  ! ринок: {type(e).__name__}: {e}")
            now = datetime.now(timezone.utc)
            nxt = now.replace(minute=minute, second=30, microsecond=0)
            if nxt <= now:
                nxt += timedelta(hours=1)
            time.sleep((nxt - now).total_seconds())
    th = threading.Thread(target=run, daemon=True, name="market")
    th.start()
    return th


# ---------------------------------------------------------------- читання: дашборд, журнал, бектест

_cache: dict = {}


def frame(conn, max_age: float = 60.0) -> pd.DataFrame:
    """Уся таблиця (кілька тисяч рядків), кеш на хвилину: журнал питає для кожного тікера."""
    c = _cache.get("frame")
    if c is not None and time.time() - c[0] < max_age:
        return c[1]
    try:
        rows = conn.execute(f"SELECT ts, {', '.join(COLS)} FROM market_breadth ORDER BY ts").fetchall()
    except Exception:
        conn.rollback()
        rows = []
    df = pd.DataFrame(rows, columns=["ts", *COLS])
    if len(df):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    _cache["frame"] = (time.time(), df)
    return df


def at(df: pd.DataFrame, moment) -> pd.Series | None:
    """Стан ринку, відомий у момент moment: остання година, що закрилась до нього (не старша за STALE)."""
    if df.empty:
        return None
    known = df.index + H
    i = known.searchsorted(pd.Timestamp(moment), side="right") - 1
    if i < 0 or pd.Timestamp(moment) - known[i] > STALE:
        return None
    return df.iloc[i]


def allow_fn(conn, max_breadth: float, bar_minutes: int = 15):
    """Фільтр режиму «ринок не гарячий» для risk.position_state: сигнал бару ts береться, якщо на закритті бару частка
    альтів у плюсі за 7 днів ≤ max_breadth. Невідомий стан (немає даних, старші за 3 год) — береться."""
    df = frame(conn)
    bar = pd.Timedelta(minutes=bar_minutes)

    def allow(ts) -> bool:
        r = at(df, pd.Timestamp(ts) + bar)
        return r is None or not np.isfinite(r["breadth_7d"]) or r["breadth_7d"] <= max_breadth
    return allow


def both(a, b):
    """Два фільтри разом (OB і ринок): сигнал береться, лише якщо пропускають обидва."""
    if a is None or b is None:
        return a or b
    return lambda ts: a(ts) and b(ts)


def strategy_allow(conn, strat: dict | None, bar_minutes: int = 15):
    """Фільтр ринку для режиму зі strategies.json (ключ market_max) або None."""
    mx = (strat or {}).get("market_max")
    return allow_fn(conn, float(mx), bar_minutes) if mx is not None else None


def api(conn, days: int = 7) -> dict:
    """Для кнопки «Market» у дашборді: останній стан, метрики, історія за days днів."""
    df = frame(conn, max_age=30)
    if df.empty:
        return {"state": "unknown", "error": "немає даних (market.py ще не запускався)"}
    last = df.iloc[-1]
    known_at = df.index[-1] + H
    hist = df[df.index >= df.index[-1] - pd.Timedelta(days=days)]
    num = lambda v: None if v is None or pd.isna(v) else float(v)   # noqa: E731
    return {
        "state": state_of(num(last["breadth_7d"])), "ts": int(known_at.timestamp()),
        "stale": bool(pd.Timestamp(datetime.now(timezone.utc)) - known_at > STALE),
        "metrics": {c: num(last[c]) for c in COLS}, "n": int(last["n"]) if pd.notna(last["n"]) else None,
        "hot": HOT, "cold": COLD, "backtest": BACKTEST,
        "history": {"t": [int((t + H).timestamp()) for t in hist.index],
                    **{c: [num(v) for v in hist[c]] for c in ("breadth_7d", "breadth_24h", "alt_med_24h", "btc_ret_24h")}},
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Стан ринку альтів: частка альтів у плюсі за 7 днів / 24 год, BTC → market_breadth")
    ap.add_argument("--show", action="store_true", help="лише показати останній стан, нічого не завантажувати")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        if not args.show:
            update(conn)
        a = api(conn)
        if a.get("error"):
            sys.exit(a["error"])
        m = a["metrics"]
        print(f"Стан на {datetime.fromtimestamp(a['ts'], timezone.utc):%Y-%m-%d %H:%M} UTC: {a['state']}"
              f"{' (застаріло)' if a['stale'] else ''} · альтів {a['n']}")
        for c in COLS[1:]:
            v = m.get(c)
            print(f"  {c:<15} {'—' if v is None else f'{v:+.3f}'}")


if __name__ == "__main__":
    main()
