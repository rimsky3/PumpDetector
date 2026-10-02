"""
Профіль обсягу (volume profile) за проміжок часу: обсяг на кожному рівні ціни, POC, VAH/VAL.

Спільний для графіків дашборду й розмітки (/api/profile, app/js/profile.js) і для експериментів з рівнями.
Будується зі свічок (не з тиків, як ATAS): обсяг кожної свічки розподіляється рівномірно по її діапазону
low–high. Береться найдрібніший ТФ, що покриває проміжок: 5m → 15m → 1h. На профілі дня чи кількох годин з 5m
POC збігається з тиковим з точністю до 1–2 рівнів; фрагмент на 30–60 хв з 5m — грубий.
Обсяг — сума Binance + Bybit у USD (як на графіку); частка покупок — з ринків, де є taker buy (у 5m/15m — Binance).

  python volume_profile.py SOON --days 2          # профілі днів за 2 дні в консоль
  python volume_profile.py SOON --from "2026-10-01 08:00" --to "2026-10-01 14:00"
"""
import argparse
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

VALUE_AREA = 0.70            # частка обсягу в value area (стандарт Market Profile)
ROWS_RANGE = 60              # рівнів на фрагмент (крок — «круглий», від діапазону ціни)
ROWS_PERIOD = 40             # рівнів на типовий день / тиждень (крок спільний для всіх профілів запиту)
SOURCES = [("5m", "5min", "5min", 300), ("15m", "15min", "15min", 900), ("1h", "1hour", "h", 3600)]
PERIODS = ("day", "week")


def nice_step(x: float) -> float:
    """Найближчий «круглий» крок ≥ x: 1, 2, 2.5, 5 × 10^k (як шкала ціни)."""
    if not x or x <= 0 or not math.isfinite(x):
        return 0.0
    p = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * p >= x * (1 - 1e-9):
            return m * p
    return 10 * p


def build(bars: pd.DataFrame, step: float, share: float = VALUE_AREA) -> dict | None:
    """Профіль з барів (high, low, volume[, buy_share]). Рівень k — ціни [k·step, (k+1)·step)."""
    b = bars.dropna(subset=["high", "low", "volume"])
    b = b[b["volume"] > 0]
    if b.empty or step <= 0:
        return None
    lo_k = np.floor(b["low"].to_numpy() / step).astype(np.int64)
    hi_k = np.floor(b["high"].to_numpy() / step).astype(np.int64)
    k0, k1 = int(lo_k.min()), int(hi_k.max())
    n = k1 - k0 + 1
    vol, buy = np.zeros(n), np.zeros(n)
    lows, highs, vols = b["low"].to_numpy(), b["high"].to_numpy(), b["volume"].to_numpy()
    bs = b["buy_share"].to_numpy() if "buy_share" in b else np.full(len(b), np.nan)
    has_buy = False
    for lo, hi, v, s, a, z in zip(lows, highs, vols, bs, lo_k, hi_k):
        if z == a or hi <= lo:                       # свічка в одному рівні
            w = np.array([v])
        else:                                        # частка діапазону свічки, що припадає на кожен рівень
            edges = np.arange(a, z + 2) * step
            ov = np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo)
            w = v * np.clip(ov, 0, None) / (hi - lo)
        vol[a - k0:z - k0 + 1] += w
        if s == s:                                   # не NaN
            buy[a - k0:z - k0 + 1] += w * s
            has_buy = True
    poc = int(np.argmax(vol))
    # value area: від POC додаємо сусідній рівень з більшим обсягом, поки не набереться share
    lo_i = hi_i = poc
    acc, total = vol[poc], vol.sum()
    while acc < share * total and (lo_i > 0 or hi_i < n - 1):
        up = vol[hi_i + 1] if hi_i < n - 1 else -1
        dn = vol[lo_i - 1] if lo_i > 0 else -1
        if up >= dn:
            hi_i += 1
            acc += up
        else:
            lo_i -= 1
            acc += dn
    return {
        "step": step, "k0": k0,
        "vol": [round(float(x), 2) for x in vol],
        "buy": [round(float(x), 2) for x in buy] if has_buy else None,
        "poc": (k0 + poc + 0.5) * step,
        "vah": (k0 + hi_i + 1) * step, "val": (k0 + lo_i) * step,
        "total": float(total),
    }


def load_bars(conn, base: str, t_from: pd.Timestamp, t_to: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    """Бари [t_from, t_to) найдрібнішого ТФ, що покриває проміжок (5m → 15m → 1h)."""
    from features import INTERVAL, aggregate, load_raw   # важкий імпорт — лише коли потрібні дані
    fallback = (pd.DataFrame(), "")
    for name, interval, freq, sec in SOURCES:
        ex = "all" if interval == INTERVAL else None     # 1h — Coinalyze (усі біржі), 5m/15m — біржі графіка
        raw = load_raw(conn, [base], interval, since=(t_from - pd.Timedelta(seconds=sec)).to_pydatetime(), exchanges=ex)
        h = aggregate(raw, base, freq)
        if h.empty:
            continue
        h = h[(h.index >= t_from) & (h.index < t_to)]
        if h.empty:
            continue
        h = h.assign(buy_share=(h["buy_volume"] / h["volume_bv"]).where(h["volume_bv"] > 0) if "volume_bv" in h else np.nan)
        cols = ["high", "low", "volume", "buy_share"]
        if h.index.min() <= t_from + pd.Timedelta(seconds=sec):   # дані покривають початок проміжку
            return h[cols], name
        if fallback[0].empty:
            fallback = (h[cols], name)
    return fallback


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(float(x), unit="s", tz="UTC") if str(x).replace(".", "", 1).isdigit() else pd.Timestamp(x, tz="UTC")


def api(conn, q: dict) -> dict:
    """/api/profile?symbol=&from=&to=&mode=range|day|week. from/to — секунди UTC; to — виключно.
    range — один профіль; day / week — профіль на кожен день (тиждень з понеділка) UTC, крок спільний."""
    base = q.get("symbol", "").upper()
    if not base:
        raise ValueError("не вказано тікер")
    t_from, t_to = _ts(q["from"]), _ts(q["to"])
    if t_to <= t_from:
        raise ValueError("порожній проміжок")
    mode = q.get("mode", "range")
    if mode in PERIODS:                     # повні періоди, що перетинаються з проміжком
        t_from = t_from.floor("D") if mode == "day" else (t_from.floor("D") - pd.Timedelta(days=t_from.weekday()))
    bars, src = load_bars(conn, base, t_from, t_to)
    if bars.empty:
        return {"src": None, "profiles": []}
    if mode not in PERIODS:
        step = nice_step((bars["high"].max() - bars["low"].min()) / ROWS_RANGE)
        p = build(bars, step)
        return {"src": src, "profiles": [{**p, "from": t_from.timestamp(), "to": t_to.timestamp()}] if p else []}
    day = bars.index.floor("D")
    groups = bars.groupby(day - pd.to_timedelta(bars.index.weekday, unit="D") if mode == "week" else day)
    ranges = [g["high"].max() - g["low"].min() for _, g in groups]
    step = nice_step(float(np.median(ranges)) / ROWS_PERIOD) if ranges else 0
    out = []
    for start, g in groups:
        p = build(g, step)
        if p:
            end = start + (pd.Timedelta(days=1) if mode == "day" else pd.Timedelta(days=7))
            out.append({**p, "from": start.timestamp(), "to": end.timestamp()})
    return {"src": src, "profiles": out}


def main() -> None:
    import psycopg
    from dotenv import load_dotenv
    ap = argparse.ArgumentParser(description="Профіль обсягу тікера: POC, VAH/VAL за днями або за проміжком.")
    ap.add_argument("ticker", help="базовий актив, напр. SOON")
    ap.add_argument("--days", type=float, default=1, help="профілі днів за останні N днів (якщо не задано --from)")
    ap.add_argument("--from", dest="frm", help="початок проміжку, UTC «YYYY-MM-DD HH:MM» — один профіль")
    ap.add_argument("--to", help="кінець проміжку (виключно), UTC; за замовчуванням — зараз")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    now = pd.Timestamp(datetime.now(timezone.utc))
    to = pd.Timestamp(args.to, tz="UTC") if args.to else now
    if args.frm:
        q = {"symbol": args.ticker, "from": pd.Timestamp(args.frm, tz="UTC").timestamp(), "to": to.timestamp(), "mode": "range"}
    else:
        q = {"symbol": args.ticker, "from": (to - pd.Timedelta(days=args.days)).timestamp(), "to": to.timestamp(), "mode": "day"}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        res = api(conn, q)
    print(f"{args.ticker.upper()}: дані {res['src'] or '—'}, профілів {len(res['profiles'])}")
    for p in res["profiles"]:
        a = pd.Timestamp(p["from"], unit="s", tz="UTC").strftime("%m-%d %H:%M")
        z = pd.Timestamp(p["to"], unit="s", tz="UTC").strftime("%m-%d %H:%M")
        print(f"  {a} → {z}  POC {p['poc']:.6g}  VA {p['val']:.6g}–{p['vah']:.6g}  крок {p['step']:g}  "
              f"рівнів {len(p['vol'])}  обсяг {p['total'] / 1e6:.1f} млн $")


if __name__ == "__main__":
    main()
