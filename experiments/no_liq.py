"""
experiments/no_liq.py — моделі без ліквідацій (2026-10-07).

Питання: чи потрібні ліквідації на НОВІЙ розмітці (імпульси)? На старій (29.09) без них LightGBM падав 0.71 → 0.60,
але відтоді їхня частка gain упала з 18.5% до 3–5% (розмітка описує старт, а не шорт-сквіз). Якщо без них якість та сама,
15m-оцінка може не чекати Coinalyze (~200 викликів 15m-ліквідацій щоцикла — головна затримка сигналу), а звірка власних
ліквідацій з Coinalyze (~20.10) стає непотрібною.

Варіанти (обидві моделі свіжонавчені на поточній розмітці, щоб різниця була лише в ознаках):
    base      15m FEATURES_15M + 1h FEATURES
    noliq     15m і 1h без ознак ліквідацій (liq_*)
    noliq15   15m без ліквідацій, 1h — base (проміжний: годинне підтвердження з ліквідаціями)
Перевірка: leave-ticker-out (15m «Кандидат», 1h — події) і бектест на монетах бектесту поза навчанням (Alert = 15m + 1h
≥ 0.75 порогу, годинні ознаки — з власних 15m, як у скринері), поточний і частковий вихід, стоп 2.48 ATR.

    python experiments/no_liq.py
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import exit_partial as P
import train
from features import (FEATURES, FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, build_hourly_own, load_pumps, load_raw,
                      query_df, short_features)
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

DAYS, K = 30, 2.48
OUT = os.path.join("reports", "no_liq.md")
NOLIQ = lambda cols: [c for c in cols if not c.startswith("liq_")]   # noqa: E731
VARIANTS = {"base": (FEATURES_15M, FEATURES), "noliq": (NOLIQ(FEATURES_15M), NOLIQ(FEATURES)),
            "noliq15": (NOLIQ(FEATURES_15M), FEATURES)}


def pf(v: pd.Series) -> str:
    gl = -v[v <= 0].sum()
    return f"{v[v > 0].sum() / gl:.2f}" if gl else "∞"


def stats(df: pd.DataFrame) -> dict:
    v = df["ret"].sort_values()
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "PF": pf(v), "PF без топ-3": pf(v.iloc[:-3]),
            "PF ≥ 20 млн": pf(df[df["band"] == "≥ 20 млн"]["ret"]), "PF 5–20 млн": pf(df[df["band"] == "5–20 млн"]["ret"])}


def ev_row(d, signal, pumps, bar) -> dict:
    ev = train.event_metrics(d, signal, pumps, bar)
    return {"спіймано": ev["pumps_caught"], "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
            "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| " + str(i) + " | " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="15m- і годинна модель без ознак ліквідацій: LTO і бектест на нових монетах").parse_args()
    load_dotenv(".env")
    excluded = excluded_tickers(".")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")} - excluded
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        pumps = pumps[~pumps["base_asset"].isin(excluded)]
        bases = sorted(reviewed | set(pumps["base_asset"]))
        d = build_from_raw(load_raw(conn, bases, INTERVAL_15M), pumps, bars_per_hour=4, extra=short_features)
        d1 = query_df(conn, "SELECT * FROM features ORDER BY base_asset, ts")
    # навчальна вибірка — як train.py: переглянуті тікери цілком + розмічені інтервали інших; свіжі нерозмічені — геть
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()].reset_index(drop=True)
    d1 = d1[(d1["base_asset"].isin(reviewed) | d1["pattern_type"].notna()) & ~d1["base_asset"].isin(excluded)]
    # 15m-історія тікера — не раніше за його 1h (як train.py, from_1h_start)
    first1 = d1.groupby("base_asset")["ts"].min()
    d = d[d["ts"] >= d["base_asset"].map(first1)].reset_index(drop=True)
    d1 = d1[(d1["ts"] <= cutoff) | d1["pattern_type"].notna()].reset_index(drop=True)
    y, groups = d["label"].astype(int), d["base_asset"]
    y1, groups1 = d1["label"].astype(int), d1["base_asset"]
    trained = set(groups) | set(groups1)
    print(f"15m: {len(d)} барів, 1h: {len(d1)} годин, тікерів {groups.nunique()}", flush=True)

    lto15, lto1, models15, models1 = {}, {}, {}, {}
    for cols15 in {tuple(v[0]) for v in VARIANTS.values()}:
        name = "з ліквідаціями" if cols15 == tuple(FEATURES_15M) else "без ліквідацій"
        print(f"  15m {name}: {len(cols15)} ознак, leave-ticker-out…", flush=True)
        p = train.oof_proba(train.make_lightgbm, d[list(cols15)], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        lto15[name] = ev_row(d, p >= thr, pumps, train.BAR_15M)
        models15[cols15] = (train.make_lightgbm().fit(d[list(cols15)], y), thr)
    for cols1 in {tuple(v[1]) for v in VARIANTS.values()}:
        name = "з ліквідаціями" if cols1 == tuple(FEATURES) else "без ліквідацій"
        print(f"  1h {name}: {len(cols1)} ознак, leave-ticker-out…", flush=True)
        p = train.oof_proba(train.make_lightgbm, d1[list(cols1)], y1, groups1)
        thr = train.best_f1_threshold(y1.to_numpy(), p)
        lto1[name] = ev_row(d1, p >= thr, pumps, pd.Timedelta(hours=1))
        models1[cols1] = (train.make_lightgbm().fit(d1[list(cols1)], y1), thr)

    coins, band = set(), {}
    for path in glob.glob(os.path.join("reports", "backtests", "20261001_*_alert.json")):
        meta = json.load(open(path, encoding="utf-8"))["meta"]
        for u in meta["universe"]:
            coins.add(u["base"])
            band[u["base"]] = "≥ 20 млн" if meta["min_turnover"] >= 20e6 else "5–20 млн"
    coins = sorted(coins - trained)
    start = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=DAYS)
    since = start - pd.Timedelta(days=9)
    res = {f"{n} · {e}": [] for n in VARIANTS for e in ("поточний вихід", "частковий вихід")}
    print(f"  бектест: {len(coins)} монет поза навчанням, {start:%m-%d}…", flush=True)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for n, b in enumerate(coins, 1):
            raw = load_raw(conn, [b], INTERVAL_15M, since=since)
            f15 = build_from_raw(raw, None, bars_per_hour=4, extra=short_features)
            h1 = build_hourly_own(conn, [b], since=since)
            if f15.empty or h1.empty:
                continue
            h = aggregate(raw, b, "15min")
            h = h[h.index >= start - pd.Timedelta(days=1)]
            for name, (c15, c1) in VARIANTS.items():
                mo15, thr15 = models15[tuple(c15)]
                mo1, thr1 = models1[tuple(c1)]
                s15 = f15[["base_asset", "ts"]].assign(probability=mo15.predict_proba(f15[c15])[:, 1], threshold=thr15)
                s1 = h1[["base_asset", "ts"]].assign(probability=mo1.predict_proba(h1[c1])[:, 1], threshold=thr1)
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                sc = t.loc[t.index >= start, "alert_score"]
                res[f"{name} · поточний вихід"] += [{**x, "band": band.get(b, "")} for x in P.simulate(h, sc, K, exit_bars, None, 0, False)]
                res[f"{name} · частковий вихід"] += [{**x, "band": band.get(b, "")} for x in P.simulate(h, sc, K, exit_bars, 3.0, 0.5, True)]
            if n % 40 == 0:
                print(f"    {n}/{len(coins)}", flush=True)

    t15, t1 = pd.DataFrame(lto15).T, pd.DataFrame(lto1).T
    bt = pd.DataFrame({k: stats(pd.DataFrame(v)) for k, v in res.items()}).T
    lines = ["# Моделі без ліквідацій", "", f"Навчання: {groups.nunique()} тікерів, {int((pumps['pattern_type'] == 'pump').sum())} імпульсів "
             f"(розмітка на {datetime.now(timezone.utc):%Y-%m-%d}). Обидві моделі свіжонавчені однаково; різниця — лише ознаки liq_*.", "",
             "## 1. Leave-ticker-out, 15m (рівень «Кандидат»)", "", md(t15), "",
             "## 2. Leave-ticker-out, 1h (годинна модель окремо)", "", md(t1), "",
             f"## 3. Бектест: {len(coins)} монет поза навчанням, {DAYS} днів (Alert = 15m + 1h ≥ {share} порогу; стоп {K} ATR)", "", md(bt), ""]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {OUT}")


if __name__ == "__main__":
    main()
