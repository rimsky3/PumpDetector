"""
experiments/rolling_confirm.py — чи можна пришвидшити Alert: годинне підтвердження на КОВЗНІЙ годині замість календарної.

Зараз Alert = 15m-модель ≥ порогу І годинний LightGBM останньої ЗАКРИТОЇ календарної години ≥ 0.75 свого порогу.
Календарна година відстає в середньому на ~30 хв: о 10:45 підтвердження ще спирається на 9:00–10:00.
Варіанти (15m-оцінки — out-of-fold з models/oof_15m.parquet; годинна модель — та сама models/lightgbm.joblib для всіх,
тож порівняння між варіантами рівне, хоч для тікерів навчання годинна частина in-sample):

    cand          лише Кандидат (15m, без підтвердження)
    cal_coinalyze як зараз: календарна година, годинні дані Coinalyze
    cal_own       календарна година з наших 15m-даних (Binance+Bybit+ліквідації) — щоб відділити ефект джерела
    roll_own      ковзна година з наших 15m-даних: кожні 15 хв — година, що закінчилась із закриттям 15m-бару
    roll_own_060  те саме, поріг підтвердження 0.6 замість 0.75
    cal_060       календарна Coinalyze, поріг 0.6

Метрики (train.event_metrics на 15m-сітці): спіймано, медіанна затримка, невдалі імпульси з сигналом, хибні епізоди
на 30 днів; «пройдено руху» на першому сигналі (медіана, як у early_phase.py).

    python experiments/rolling_confirm.py
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
from features import FEATURES, INTERVAL_15M, aggregate, build_from_raw, in_pump, load_pumps, load_raw

B = pd.Timedelta(minutes=15)


def hourly_from_15m(raw: dict, offset_min: int) -> dict:
    """15m-ряди контрактів → години, що починаються на :offset (0/15/30/45). ts — відкриття такої «години»."""
    keys = ["base_asset", "market", "exchange", "quote_asset"]
    off = pd.Timedelta(minutes=offset_min)

    def roll(df, agg, extra=()):
        if df.empty:
            return df
        d = df.assign(ts=(df["ts"] - off).dt.floor("h") + off)
        r = d.groupby(keys + list(extra) + ["ts"], dropna=False).agg(**agg, n=("ts", "size")).reset_index()
        return r[r["n"] == 4].drop(columns="n")

    out = {
        "ohlcv": roll(raw["ohlcv"], dict(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"),
                                         bv=("bv", lambda x: x.sum(min_count=1))), ["volume_denominated"]),
        "oi": roll(raw["oi"], dict(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))),
        "funding": roll(raw["funding"], dict(c=("c", "last"))),
        "ls": roll(raw["ls"], dict(long_pct=("long_pct", "last"))),
        "liq": roll(raw["liq"], dict(long_liq=("long_liq", "sum"), short_liq=("short_liq", "sum"))),
    }
    if "liq_coverage" in raw:
        out["liq_coverage"] = raw["liq_coverage"]
    return out


def moved(d: pd.DataFrame, sig: np.ndarray, pumps: pd.DataFrame, prices: dict) -> list[float]:
    out = []
    s = d.loc[sig, ["base_asset", "ts"]]
    for p in pumps[pumps["pattern_type"] == "pump"].itertuples():
        h = prices.get(p.base_asset)
        if h is None:
            continue
        ss = s[(s["base_asset"] == p.base_asset) & in_pump(s["ts"], p.start_ts, p.end_ts, B)]
        before = h[h.index + B <= p.start_ts]
        inside = h[(h.index >= p.start_ts) & (h.index < p.end_ts)]
        if ss.empty or before.empty or inside.empty:
            continue
        p0, pk = before["price"].iloc[-1], inside["price"].max()
        ps = h["price"].get(ss["ts"].min())
        if ps is not None and pk > p0:
            out.append(float(np.clip((ps - p0) / (pk - p0), 0, 1)))
    return out


def main() -> None:
    argparse.ArgumentParser(description="Ковзна година для підтвердження Alert проти календарної (експеримент)").parse_args()
    load_dotenv(".env")
    oof = pd.read_parquet(os.path.join(train.MODELS_DIR, "oof_15m.parquet"))
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    m1 = joblib.load(train.model_path("lightgbm"))
    thr1 = m1["threshold"]
    bases = sorted(oof["base_asset"].unique())
    s15 = (oof["p15"] / oof["thr15"]).to_numpy()
    ctx_cal = (oof["ts"] + B - pd.Timedelta(hours=1)).dt.floor("h")      # остання закрита календарна година
    ctx_roll = oof["ts"] + B - pd.Timedelta(hours=1)                      # година, що закрилась разом із баром

    def score(df):
        return df[["base_asset", "ts"]].assign(s1h=m1["model"].predict_proba(df[FEATURES])[:, 1] / thr1)

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        print("годинні ознаки: Coinalyze…")
        cal_c = score(build_from_raw(load_raw(conn, bases)))
        print("годинні ознаки з 15m: 4 зсуви години…")
        raw15 = load_raw(conn, bases, INTERVAL_15M)
        own = {k: score(build_from_raw(hourly_from_15m(raw15, k))) for k in (0, 15, 30, 45)}
        prices = {b: aggregate(raw15, b, "15min") for b in bases}
    cal_o = own[0]
    roll_o = pd.concat(own.values(), ignore_index=True)

    def ctx(table, key):
        k = oof[["base_asset"]].assign(ts=key)
        return k.merge(table, on=["base_asset", "ts"], how="left")["s1h"].fillna(0).to_numpy()

    variants = {
        "cand": s15 >= 1,
        "cal_coinalyze": (s15 >= 1) & (ctx(cal_c, ctx_cal) >= 0.75),
        "cal_own": (s15 >= 1) & (ctx(cal_o, ctx_cal) >= 0.75),
        "roll_own": (s15 >= 1) & (ctx(roll_o, ctx_roll) >= 0.75),
        "roll_own_060": (s15 >= 1) & (ctx(roll_o, ctx_roll) >= 0.60),
        "cal_060": (s15 >= 1) & (ctx(cal_c, ctx_cal) >= 0.60),
    }
    d = oof[["base_asset", "ts"]]
    rows = {}
    for name, sig in variants.items():
        ev = train.event_metrics(d, sig, pumps, B)
        mv = moved(d, sig, pumps, prices)
        rows[name] = {"спіймано": ev["pumps_caught"],
                      "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                      "пройдено руху": f"{np.median(mv):.0%}" if mv else "—",
                      "невдалі з сигналом": ev["false_starts_triggered"],
                      "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
    print("\n" + pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
