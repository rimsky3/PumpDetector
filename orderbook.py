"""
orderbook.py — співвідношення біди / аски в стакані ±2% від ціни (Binance + Bybit), наживо й історія знімків.

Навіщо: experiments/book_spot.py і entry_rules_backtest.py — коли в стакані ±2% продавців більше, ніж покупців (біди / аски < 1),
входи моделі гірші (PF 0.76 проти 0.98); разом з лімітним входом −0.5 ATR — найкраща конфігурація (PF 1.08). У дашборді це
«OB» біля Alert і картка «Біди / аски ±2%»; скринер пише знімки в таблицю book_ratio, щоб мати значення в момент сигналу
(і чесну перевірку правила на live-сигналах).

    ratio = (Σ бідів від найкращого вниз до mid·0.98) / (Σ асків від найкращого вгору до mid·1.02), у $, сума двох бірж.
    ≥ 1 — покупців у стакані не менше, ніж продавців.

Джерела (публічні, без ключа): Binance GET /fapi/v1/depth (limit 1000), Bybit GET /v5/market/orderbook (linear, limit 500).

    python orderbook.py GTC MOVR      # поточні значення (перевірка), без запису
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

BAND = 0.02                     # смуга ±2% від середини спреду — та сама, що в бектесті
EXCHANGES = ("Binance", "Bybit")
_S = requests.Session()


def _side_usd(levels, mid: float, side: str) -> tuple[float, bool]:
    """Сума $ рівнів у смузі; True — книжка сягає межі смуги (інакше глибини не вистачило й сума неповна)."""
    lim = mid * (1 - BAND) if side == "bid" else mid * (1 + BAND)
    total, reached = 0.0, False
    for p, q in levels:
        p, q = float(p), float(q)
        if (side == "bid" and p < lim) or (side == "ask" and p > lim):
            reached = True
            break
        total += p * q
    return total, reached


def fetch(exchange: str, symbol: str) -> dict | None:
    """Біди/аски ±2% одного ринку. None — помилка або порожній стакан."""
    try:
        if exchange == "Binance":
            r = _S.get("https://fapi.binance.com/fapi/v1/depth", params={"symbol": symbol, "limit": 1000}, timeout=8).json()
            bids, asks = r.get("bids") or [], r.get("asks") or []
        else:
            r = _S.get("https://api.bybit.com/v5/market/orderbook",
                       params={"category": "linear", "symbol": symbol, "limit": 500}, timeout=8).json()
            res = r.get("result") or {}
            bids, asks = res.get("b") or [], res.get("a") or []
    except (requests.RequestException, ValueError):
        return None
    if not bids or not asks:
        return None
    mid = (float(bids[0][0]) + float(asks[0][0])) / 2
    b, b_full = _side_usd(bids, mid, "bid")
    a, a_full = _side_usd(asks, mid, "ask")
    return {"bid": b, "ask": a, "full": b_full and a_full}


def markets(conn, bases: list[str]) -> dict[str, list[tuple[str, str]]]:
    """{тікер: [(біржа, символ на біржі), …]} — USDT-перпи Binance і Bybit з таблиці markets."""
    out: dict[str, list[tuple[str, str]]] = {}
    for base, ex, sym in conn.execute("""SELECT base_asset, exchange_name, symbol_on_exchange FROM markets
                                         WHERE base_asset = ANY(%s) AND exchange_name = ANY(%s) AND quote_asset = 'USDT'""",
                                      (bases, list(EXCHANGES))):
        out.setdefault(base, []).append((ex, sym))
    return out


def snapshot(mk: list[tuple[str, str]]) -> dict | None:
    """Знімок одного тікера: сума по біржах + співвідношення кожної біржі окремо."""
    parts = {ex: fetch(ex, sym) for ex, sym in mk}
    ok = {ex: p for ex, p in parts.items() if p and p["ask"] > 0}
    if not ok:
        return None
    bid, ask = sum(p["bid"] for p in ok.values()), sum(p["ask"] for p in ok.values())
    return {"ratio": bid / ask if ask else None, "bid_usd": bid, "ask_usd": ask,
            "binance": ok["Binance"]["bid"] / ok["Binance"]["ask"] if "Binance" in ok else None,
            "bybit": ok["Bybit"]["bid"] / ok["Bybit"]["ask"] if "Bybit" in ok else None,
            "full": all(p["full"] for p in ok.values()), "ts": datetime.now(timezone.utc)}


def ensure_table(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS book_ratio (
        base_asset TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL, ratio DOUBLE PRECISION, bid_usd DOUBLE PRECISION,
        ask_usd DOUBLE PRECISION, binance DOUBLE PRECISION, bybit DOUBLE PRECISION, PRIMARY KEY (base_asset, ts))""")
    conn.commit()


def record(conn, bases: list[str]) -> int:
    """Знімки для тікерів (паралельно) → таблиця book_ratio. Повертає кількість записаних."""
    if not bases:
        return 0
    ensure_table(conn)
    mk = markets(conn, bases)
    with ThreadPoolExecutor(8) as pool:
        snaps = dict(zip(mk, pool.map(snapshot, mk.values())))
    rows = [(b, s["ts"], s["ratio"], s["bid_usd"], s["ask_usd"], s["binance"], s["bybit"]) for b, s in snaps.items() if s]
    conn.cursor().executemany("""INSERT INTO book_ratio (base_asset, ts, ratio, bid_usd, ask_usd, binance, bybit)
                                 VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""", rows)
    conn.commit()
    return len(rows)


def history(conn, base: str, since) -> list[tuple[datetime, float]]:
    ensure_table(conn)
    return conn.execute("SELECT ts, ratio FROM book_ratio WHERE base_asset = %s AND ts >= %s ORDER BY ts",
                        (base, since)).fetchall()


def at(rows: list[tuple[datetime, float]], start, end) -> float | None:
    """Перший знімок у вікні [start, end] (момент сигналу: скринер пише за кілька хвилин після закриття бару)."""
    for ts, r in rows:
        if start <= ts <= end:
            return r
        if ts > end:
            break
    return None


def at_signal(rows: list[tuple[datetime, float]], ts, bar_minutes: int = 15) -> float | None:
    """OB у момент сигналу: перший знімок за ≤ 12 хв після закриття бару сигналу (скринер пише після оцінки)."""
    import pandas as pd
    close = (pd.Timestamp(ts) + pd.Timedelta(minutes=bar_minutes)).to_pydatetime()
    return at(rows, close, close + timedelta(minutes=12))


def allow_fn(rows: list[tuple[datetime, float]], bar_minutes: int = 15):
    """Фільтр режиму «OB ≥ 1» для risk.position_state: невідомий OB (немає знімка) — сигнал береться."""
    def allow(ts) -> bool:
        v = at_signal(rows, ts, bar_minutes)
        return v is None or v >= 1
    return allow


# ---------------------------------------------------------------- історія з архіву Binance (бектест)

ARCHIVE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "book_cache")


def archive_day(symbol: str, day: str):
    """Денний файл bookDepth Binance (кожні 30 с, смуги ±0.2…5%): кеш data/book_cache/ (той самий, що в
    experiments/book_spot.py). Порожній DataFrame — файлу немає."""
    import io
    import zipfile
    from urllib.parse import quote
    import pandas as pd
    os.makedirs(ARCHIVE_CACHE, exist_ok=True)
    path = os.path.join(ARCHIVE_CACHE, f"{symbol}-bookDepth-{day}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{quote(symbol)}/{quote(symbol)}-bookDepth-{day}.zip"
    df = pd.DataFrame(columns=["timestamp", "percentage", "notional"])
    for attempt in range(4):
        try:
            r = _S.get(url, timeout=60)
            if r.status_code == 404:
                break
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                df = pd.read_csv(z.open(z.namelist()[0]), usecols=["timestamp", "percentage", "notional"])
            break
        except (requests.RequestException, zipfile.BadZipFile):
            time.sleep(3 + attempt * 3)
    if len(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_parquet(path)
    return df


def archive_allow_fn(symbol: str, signal_ts: list, bar_minutes: int = 15):
    """Фільтр «OB ≥ 1» для бектесту: біди / аски ±2% з архіву Binance (лише Binance; наживо — Binance + Bybit) на
    закриття бару сигналу; немає знімка в межах 5 хв — сигнал береться."""
    import pandas as pd
    days = sorted({(pd.Timestamp(t) + pd.Timedelta(minutes=bar_minutes)).strftime("%Y-%m-%d") for t in signal_ts})
    parts = [archive_day(symbol, d) for d in days]
    parts = [p for p in parts if len(p)]
    w = pd.concat(parts).pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last").sort_index() \
        if parts else pd.DataFrame()

    def allow(ts) -> bool:
        if w.empty or -2.0 not in w.columns or 2.0 not in w.columns:
            return True
        t = pd.Timestamp(ts) + pd.Timedelta(minutes=bar_minutes)
        pre = w[w.index <= t]
        if pre.empty or t - pre.index[-1] > pd.Timedelta(minutes=5):
            return True
        b, a = pre[-2.0].iloc[-1], pre[2.0].iloc[-1]
        return not (a > 0) or b / a >= 1
    return allow


_live: dict = {}


def live(conn, base: str, max_age: float = 4.0) -> dict | None:
    """Поточний знімок для дашборду (кеш кілька секунд) + значення ~15 хв тому з таблиці; раз на хвилину — запис."""
    hit = _live.get(base)
    if hit and time.time() - hit[0] < max_age:
        return hit[1]
    mk = markets(conn, [base]).get(base)
    snap = snapshot(mk) if mk else None
    if snap:
        ensure_table(conn)
        last = conn.execute("SELECT max(ts) FROM book_ratio WHERE base_asset = %s", (base,)).fetchone()[0]
        if last is None or (snap["ts"] - last).total_seconds() >= 60:
            conn.execute("""INSERT INTO book_ratio (base_asset, ts, ratio, bid_usd, ask_usd, binance, bybit)
                            VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""",
                         (base, snap["ts"], snap["ratio"], snap["bid_usd"], snap["ask_usd"], snap["binance"], snap["bybit"]))
            conn.commit()
        ago = conn.execute("""SELECT ratio FROM book_ratio WHERE base_asset = %s AND ts BETWEEN %s AND %s
                              ORDER BY ts DESC LIMIT 1""",
                           (base, snap["ts"] - timedelta(minutes=21),
                            snap["ts"] - timedelta(minutes=12))).fetchone()
        snap = {**snap, "ratio_15m": ago[0] if ago else None}
    _live[base] = (time.time(), snap)
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(description="Біди / аски ±2% (Binance + Bybit) для тікерів — перевірка, без запису")
    ap.add_argument("tickers", nargs="+", help="базові активи, напр. GTC MOVR")
    args = ap.parse_args()
    import psycopg
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        mk = markets(conn, [t.upper() for t in args.tickers])
    for b, m in mk.items():
        s = snapshot(m)
        if not s:
            print(f"{b}: немає даних стакану")
            continue
        fmt = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{b}: OB {fmt(s['ratio'])} (біди ${s['bid_usd']:,.0f} / аски ${s['ask_usd']:,.0f}) · Binance {fmt(s['binance'])} · "
              f"Bybit {fmt(s['bybit'])}" + ("" if s["full"] else " · глибини не вистачило до ±2%"))


if __name__ == "__main__":
    main()
