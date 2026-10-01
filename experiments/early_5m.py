"""
experiments/early_5m.py — чи можна отримувати сигнал раніше: 15m-модель кожні 5 хв (ковзні 15m-бари) і рівень «Підготовка».

Зараз 15m-модель рахується на закритті 15m-барів (:00, :15, :30, :45) — від того, як модель «побачила», до сигналу до 15 хв
плюс обробка. Варіанти (модель та сама, без перенавчання; годинне підтвердження — остання закрита година з власних 15m):

    aligned      як зараз: бари 15m по :00/:15/:30/:45, Alert = 15m ≥ поріг і 1h ≥ 0.75
    rolling      15m-бари зі зсувом 0/5/10 хв (з 5m-даних бірж і 5m-ліквідацій) — оцінка кожні 5 хв
    prep         «Підготовка» на ковзних барах: 15m ≥ 0.93 порогу 2 кроки поспіль, ціна й OI за 15 хв ростуть
                 (рання тиха підказка, підтвердження годинною не потрібне)

Метрики проти розмітки (тікери з навчання — модель їх бачила, тож абсолютні цифри оптимістичні, порівнюємо ВАРІАНТИ між
собою): спіймано, затримка від розміченого старту до ЗАКРИТТЯ бару сигналу, пройдено руху на першому сигналі, невдалі
імпульси з сигналом, хибні епізоди / 30 днів. Усі сигнали — на 5-хвилинній сітці моментів закриття.

    python experiments/early_5m.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from features import (FEATURES, FEATURES_15M, aggregate, build_from_raw, build_hourly_own, in_pump, load_pumps, load_raw,
                      short_features)
from fetch import read_tickers_file
from tiers import context_hour

B5, B15 = pd.Timedelta(minutes=5), pd.Timedelta(minutes=15)


def roll15(raw: dict, offset_min: int) -> dict:
    """5m-ряди контрактів → 15m-бари, що починаються на :offset (0/5/10 у межах 15 хв); лише повні (3 бари)."""
    keys = ["base_asset", "market", "exchange", "quote_asset"]
    off = pd.Timedelta(minutes=offset_min)

    def roll(df, agg, extra=()):
        if df.empty:
            return df
        d = df.assign(ts=(df["ts"] - off).dt.floor("15min") + off)
        r = d.groupby(keys + list(extra) + ["ts"], dropna=False).agg(**agg, n=("ts", "size")).reset_index()
        return r[r["n"] == 3].drop(columns="n")
    out = {"ohlcv": roll(raw["ohlcv"], dict(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"),
                                            bv=("bv", lambda x: x.sum(min_count=1))), ["volume_denominated"]),
           "oi": roll(raw["oi"], dict(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))),
           "funding": roll(raw["funding"], dict(c=("c", "last"))), "ls": roll(raw["ls"], dict(long_pct=("long_pct", "last"))),
           "liq": roll(raw["liq"], dict(long_liq=("long_liq", "sum"), short_liq=("short_liq", "sum")))}
    if "liq_coverage" in raw:
        out["liq_coverage"] = raw["liq_coverage"]
    return out


def first_signal_stats(sig: pd.DataFrame, pumps: pd.DataFrame, prices: dict) -> tuple[list, list]:
    """Затримка (хв від старту до закриття бару першого сигналу в імпульсі) і пройдено руху."""
    delays, moved = [], []
    for p in pumps[pumps["pattern_type"] == "pump"].itertuples():
        s = sig[(sig["base_asset"] == p.base_asset) & (sig["close"] > p.start_ts) & (sig["close"] - B5 < p.end_ts)]
        if s.empty:
            continue
        c = s["close"].min()
        delays.append(max(0.0, (c - p.start_ts) / pd.Timedelta(minutes=1)))
        h = prices.get(p.base_asset)
        if h is not None:
            before = h[h.index + B5 <= p.start_ts]; inside = h[(h.index >= p.start_ts) & (h.index < p.end_ts)]
            at = h[h.index + B5 <= c]
            if len(before) and len(inside) and len(at):
                p0, pk, pc = before["price"].iloc[-1], inside["price"].max(), at["price"].iloc[-1]
                if pk > p0:
                    moved.append(float(np.clip((pc - p0) / (pk - p0), 0, 1)))
    return delays, moved


def main() -> None:
    argparse.ArgumentParser(description="15m-модель кожні 5 хв і рівень «Підготовка» (експеримент)").parse_args()
    load_dotenv(".env")
    m15 = joblib.load(train.model_path(train.M15_NAME)); m1 = joblib.load(train.model_path("lightgbm"))
    share = m15.get("confirm", {}).get("share", 0.75)
    bases = [t.upper() for t in read_tickers_file("tickers.txt")]
    rows = []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        for b in bases:
            raw5 = load_raw(conn, [b], "5min")
            if raw5["ohlcv"].empty:
                continue
            d1 = build_hourly_own(conn, [b])
            ctx = (d1[["ts"]].assign(s1h=m1["model"].predict_proba(d1[FEATURES])[:, 1] / m1["threshold"]).set_index("ts")["s1h"]
                   if not d1.empty else pd.Series(dtype=float))
            h5 = aggregate(raw5, b, "5min")
            for off in (0, 5, 10):
                f = build_from_raw(roll15(raw5, off), bars_per_hour=4, extra=short_features)
                if f.empty:
                    continue
                s15 = m15["model"].predict_proba(f[FEATURES_15M])[:, 1] / m15["threshold"]
                close = f["ts"] + B15
                c1h = context_hour(f["ts"])
                s1h = ctx.reindex(c1h).to_numpy() if len(ctx) else np.zeros(len(f))
                pr = h5["price"].reindex(f["ts"] + B15 - B5).to_numpy()            # закриття останнього 5m у барі
                pr_prev = h5["price"].reindex(f["ts"] - B5).to_numpy()
                oi_now = h5["oi"].reindex(f["ts"] + B15 - B5).to_numpy(); oi_prev = h5["oi"].reindex(f["ts"] - B5).to_numpy()
                rows.append(pd.DataFrame({"base_asset": b, "close": close.to_numpy(), "off": off, "s15": s15, "s1h": np.nan_to_num(s1h),
                                          "up": (pr > pr_prev) & (oi_now > oi_prev)}))
            globals().setdefault("PRICES", {})[b] = h5
    d = pd.concat(rows, ignore_index=True).sort_values(["base_asset", "close"]).reset_index(drop=True)
    d["alert"] = (d["s15"] >= 1) & (d["s1h"] >= share)
    # «Підготовка»: s15 ≥ 0.93 два кроки поспіль (на 5-хв сітці) і ціна та OI за 15 хв ростуть
    near = d["s15"] >= 0.93
    d["prep"] = near & near.groupby(d["base_asset"]).shift(1, fill_value=False) & d["up"]
    variants = {"aligned (як зараз)": (d["off"] == 0, "alert"), "rolling (кожні 5 хв)": (d["off"] >= 0, "alert"),
                "Кандидат aligned": (d["off"] == 0, "cand"), "Кандидат rolling": (d["off"] >= 0, "cand"),
                "Підготовка rolling": (d["off"] >= 0, "prep")}
    d["cand"] = d["s15"] >= 1
    out = {}
    for name, (mask, col) in variants.items():
        x = d[mask]
        # на 5-хв сітку: рядок = 5-хвилинний відрізок, що закінчується в close; сигнал, якщо будь-який бар із цим закриттям сигналить
        g = x.groupby(["base_asset", "close"])[col].any().reset_index()
        grid = g.assign(ts=g["close"] - B5)
        ev = train.event_metrics(grid[["base_asset", "ts"]], grid[col].to_numpy(), pumps, B5)
        delays, mv = first_signal_stats(g[g[col]], pumps, PRICES)
        days = grid.groupby("base_asset")["ts"].agg(lambda s: (s.max() - s.min()) / pd.Timedelta(days=1)).sum()
        out[name] = {"спіймано": ev["pumps_caught"], "затримка, хв (мед.)": round(np.median(delays)) if delays else None,
                     "пройдено руху": f"{np.median(mv):.0%}" if mv else "—", "невдалі з сигналом": ev["false_starts_triggered"],
                     "хибних / 30 д": round(ev["false_alarms_per_30d"] * (len(grid) * 5 / 1440) / days, 1) if days else None}
    print(pd.DataFrame(out).T.to_string())
    print("\nТікери з навчання (модель їх бачила): порівнювати варіанти між собою, не абсолютні цифри.")


if __name__ == "__main__":
    main()
