"""
wake.py — сигнал «прокидається» на 5m-барах: монета з флету почала рух (обсяг + ціна + OI), «подивись».

Не прогноз моделі і не сигнал на вхід: тихе попередження раніше за модель + тікер одразу стає «гарячим» (скринер
оцінює його кожні 5 хв на ковзних 15m-барах, а не чекає звичайного 15m-циклу). Правило і пороги — з
experiments/wakeup_5m.py (вересень, 86 монет ≥ 20 млн поза навчанням, варіант «K5 P2% + OI»): після спрацювання
Кандидат ≤ 2 год у 46% (випадковий момент — 6%), перед 37% Alert — на ~30 хв раніше; ціна далі — без переваги.

    обсяг за останні 15 хв ≥ VOL_X × норми (медіана 15-хв обсягу за попередні 7 днів, без останніх 15 хв)
    і ≥ LOCAL_X × медіани за останні 4 год (монета, що вже в русі, не «прокидається» вдруге: AIN 03.10 18:05 —
    ×188 до мертвої тижневої норми, але вдвічі МЕНШЕ за попередні 4 год; на бектесті −9% спрацювань, якість та сама)
    і ціна за 15 хв ≥ +PRICE_MIN   і   OI (ланцюговий індекс Binance + Bybit) за годину ≥ +OI_MIN;
    після спрацювання — COOLDOWN тиші для монети (одне попередження на рух).
Дані — 5m бірж з бази (features.load_raw / aggregate): ті самі для скринера (check) і графіка дашборду (history).

    python wake.py AIN --days 3      # спрацювання за останні 3 дні
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from features import aggregate, load_raw

ROOT = os.path.dirname(os.path.abspath(__file__))
EVENTS_FILE = os.path.join(ROOT, "reports", "wake_events.json")
VOL_X, PRICE_MIN, OI_MIN = 5.0, 0.02, 0.02
LOCAL_X, LOCAL_BARS = 2.0, 48            # обсяг ще й ≥ ×2 від медіани за останні 4 год (48 барів 5m)
COOLDOWN = pd.Timedelta(hours=4)
NORM_DAYS, MIN_NORM_DAYS = 7, 3          # норма обсягу — 7 днів 5m-історії, не менше 3
DOT_MINUTES = 30                         # скільки хвилин після спрацювання тікер позначено в списку дашборду
HOT_HOURS = 2                            # скільки годин після спрацювання тікер «гарячий» у скринері
KEEP_DAYS = 7
B5 = pd.Timedelta(minutes=5)
INTERVAL = "5min"


def metrics(h: pd.DataFrame) -> pd.DataFrame:
    """5m-ряд (volume, price, oi_log_idx; індекс — відкриття бару) → vol_x, ret15, oi1h і чи спрацювало правило."""
    v15 = h["volume"].fillna(0).rolling(3).sum()
    norm = v15.shift(3).rolling(NORM_DAYS * 288, min_periods=MIN_NORM_DAYS * 288).median()
    local = v15.shift(3).rolling(LOCAL_BARS, min_periods=LOCAL_BARS // 2).median()
    oi = h["oi_log_idx"].ffill() if "oi_log_idx" in h else pd.Series(np.nan, index=h.index)
    m = pd.DataFrame({"vol_x": v15 / norm.where(norm > 0), "vol_x4h": v15 / local.where(local > 0),
                      "ret15": h["price"] / h["price"].shift(3) - 1, "oi1h": np.expm1(oi - oi.shift(12))}, index=h.index)
    m["hit"] = (m["vol_x"] >= VOL_X) & (m["vol_x4h"] >= LOCAL_X) & (m["ret15"] >= PRICE_MIN) & (m["oi1h"] >= OI_MIN)
    return m


def events(h: pd.DataFrame) -> list[dict]:
    """Спрацювання з тишею COOLDOWN: [{ts — відкриття 5m-бару, vol_x, ret15, oi1h}]."""
    if h.empty:
        return []
    m = metrics(h)
    out, last = [], None
    for ts, r in m[m["hit"].fillna(False)].iterrows():
        if last is None or ts - last >= COOLDOWN:
            out.append({"ts": ts, "vol_x": float(r["vol_x"]), "vol_x4h": float(r["vol_x4h"]), "ret15": float(r["ret15"]),
                        "oi1h": float(r["oi1h"])})
            last = ts
    return out


def frame(conn, bases: list[str], since) -> dict[str, pd.DataFrame]:
    raw = load_raw(conn, bases, INTERVAL, since=since)
    out = {}
    for b in bases:
        h = aggregate(raw, b, INTERVAL)
        if not h.empty:
            out[b] = h
    return out


def history(conn, base: str, since: pd.Timestamp) -> list[dict]:
    """Спрацювання тікера з моменту since (дані беруться з запасом на 7-денну норму обсягу)."""
    h = frame(conn, [base], (since - pd.Timedelta(days=NORM_DAYS + 1)).to_pydatetime()).get(base)
    return [e for e in events(h) if e["ts"] >= since] if h is not None else []


def load_events() -> dict:
    try:
        with open(EVENTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def last_wake(now: datetime | None = None, minutes: float = DOT_MINUTES) -> dict[str, dict]:
    """Тікери, що «прокинулись» за останні minutes хвилин (момент — закриття 5m-бару): {тікер: подія}."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    out = {}
    for b, evs in load_events().items():
        if evs:
            e = evs[-1]
            at = pd.Timestamp(e["ts"]) + B5
            if now - at <= pd.Timedelta(minutes=minutes):
                out[b] = {**e, "at": at.isoformat(), "minutes": int((now - at).total_seconds() // 60)}
    return out


def check(conn, bases: list[str], log=print) -> list[str]:
    """Для скринера: хто «прокинувся» на останньому закритому 5m-барі → reports/wake_events.json; повертає нові тікери."""
    if not bases:
        return []
    now = pd.Timestamp(datetime.now(timezone.utc))
    last_bar = now.floor("5min") - B5
    frames = frame(conn, bases, (now - pd.Timedelta(days=NORM_DAYS + 1)).to_pydatetime())
    saved = load_events()
    new = []
    for b, h in frames.items():
        h = h[h.index <= last_bar]
        if h.empty or h.index[-1] != last_bar:
            continue
        ev = events(h)
        if ev and ev[-1]["ts"] == last_bar:
            prev = saved.get(b, [])
            if prev and pd.Timestamp(prev[-1]["ts"]) == last_bar:
                continue
            saved.setdefault(b, []).append({**ev[-1], "ts": last_bar.isoformat()})
            new.append(b)
    if new:
        cut = now - pd.Timedelta(days=KEEP_DAYS)
        saved = {b: [e for e in evs if pd.Timestamp(e["ts"]) >= cut] for b, evs in saved.items()}
        os.makedirs(os.path.dirname(EVENTS_FILE), exist_ok=True)
        tmp = EVENTS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({b: evs for b, evs in saved.items() if evs}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, EVENTS_FILE)
        log(f"  прокидається: {', '.join(new)}")
    return new


def main() -> None:
    import psycopg
    from dotenv import load_dotenv
    ap = argparse.ArgumentParser(description="Сигнал «прокидається» (5m: обсяг ×5, ціна +2% за 15 хв, OI +2% за год) для тікера.")
    ap.add_argument("ticker", help="базовий актив, напр. AIN")
    ap.add_argument("--days", type=float, default=3, help="за скільки останніх днів показати спрацювання")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(os.path.join(ROOT, ".env"))
    since = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=args.days)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        evs = history(conn, args.ticker.upper(), since)
    print(f"{args.ticker.upper()}: {len(evs)} спрацювань за {args.days:g} дн.")
    for e in evs:
        print(f"  {e['ts'] + B5:%m-%d %H:%M} UTC  обсяг ×{e['vol_x']:.1f}  ціна {e['ret15']:+.1%} за 15 хв  OI {e['oi1h']:+.1%} за год")


if __name__ == "__main__":
    main()
