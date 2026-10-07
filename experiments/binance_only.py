"""
experiments/binance_only.py — модель лише на даних Binance проти Binance + Bybit (2026-10-07).

Навіщо: в архіві Binance (data.binance.vision, binance_archive.py) є свічки з обсягом покупок, OI і частка лонгів кожні
5 хв і стакан — за роки. Якщо модель «лише Binance» не гірша, будь-який бектест (6, 12 місяців) рахується на тих самих
ознаках, що й наживо, і без Coinalyze. Мінус: Binance — ~55–70% OI монети, імпульс, що почався на Bybit, видно слабше.
Перед експериментом OI і частку лонгів Binance дозаповнено з архіву (у базі вони були лише за 30 днів API: серпень
навчання мав OI / L/S тільки з Bybit; архів = API до біта, зі зсувом +5 хв — binance_archive.py --check).

Варіанти (15m-моделі свіжонавчені на тих самих тікерах і розмітці, ознаки без ліквідацій — MODEL_FEATURES_15M):
    A both       15m з Binance + Bybit; годинне підтвердження — робоча годинна модель на власних 15m обох бірж
    B bin15      15m лише з Binance; підтвердження — як A (ізолює 15m-модель)
    C binance    15m і годинні ознаки лише з Binance (повністю «одна біржа»)
Перевірка: leave-ticker-out (15m «Кандидат») і бектест на монетах бектесту поза навчанням (обидві смуги обороту),
останні 30 днів до позавчора, частковий вихід, стоп 2.48 ATR.

    python experiments/binance_only.py
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv

import exit_partial as P
import train
from features import (INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, hourly_from_intraday, load_pumps,
                      load_raw, query_df, short_features)
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

DAYS, K = 30, 2.48
OUT = os.path.join("reports", "binance_only.md")
BOTH, BIN = ["Binance", "Bybit"], ["Binance"]
LAST_DAY = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).normalize()


def pf(v: pd.Series) -> str:
    gl = -v[v <= 0].sum()
    return f"{v[v > 0].sum() / gl:.2f}" if gl else "∞"


def stats(df: pd.DataFrame) -> dict:
    v = df["ret"].sort_values()
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "PF": pf(v), "PF без топ-3": pf(v.iloc[:-3]),
            "PF ≥ 20 млн": pf(df[df["band"] == "≥ 20 млн"]["ret"]), "PF 5–20 млн": pf(df[df["band"] == "5–20 млн"]["ret"])}


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| " + str(i) + " | " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for i, r in zip(df.index, df.to_numpy())]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="15m-модель лише на Binance проти Binance + Bybit: LTO і бектест").parse_args()
    load_dotenv(".env")
    excluded = excluded_tickers(".")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")} - excluded
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    data = {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        pumps = pumps[~pumps["base_asset"].isin(excluded)]
        bases = sorted(reviewed | set(pumps["base_asset"]))
        first1 = query_df(conn, "SELECT base_asset, min(ts) AS t FROM features GROUP BY 1").set_index("base_asset")["t"]
        for name, ex in (("both", BOTH), ("binance", BIN)):
            d = build_from_raw(load_raw(conn, bases, INTERVAL_15M, exchanges=ex), pumps, bars_per_hour=4, extra=short_features)
            d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
            d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()]
            data[name] = d[d["ts"] >= d["base_asset"].map(first1)].reset_index(drop=True)
    # однакові бари в обох наборах (тікер без Binance-ринку в «binance» випав би — порівнюємо на спільних)
    key = lambda d: pd.MultiIndex.from_frame(d[["base_asset", "ts"]])   # noqa: E731
    common = key(data["both"]).intersection(key(data["binance"]))
    for n in data:
        data[n] = data[n][key(data[n]).isin(common)].reset_index(drop=True)
    print(f"15m: {len(common)} спільних барів, {data['both']['base_asset'].nunique()} тікерів", flush=True)
    trained = set(data["both"]["base_asset"])

    lto, models = {}, {}
    for name, d in data.items():
        y, groups = d["label"].astype(int), d["base_asset"]
        print(f"  15m {name}: leave-ticker-out…", flush=True)
        p = train.oof_proba(train.make_lightgbm, d[MODEL_FEATURES_15M], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        ev = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        lto[name] = {"спіймано": ev["pumps_caught"], "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                     "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
        models[name] = (train.make_lightgbm().fit(d[MODEL_FEATURES_15M], y), thr)

    coins, band = set(), {}
    for path in glob.glob(os.path.join("reports", "backtests", "20261001_*_alert.json")):
        meta = json.load(open(path, encoding="utf-8"))["meta"]
        for u in meta["universe"]:
            coins.add(u["base"])
            band[u["base"]] = "≥ 20 млн" if meta["min_turnover"] >= 20e6 else "5–20 млн"
    coins = sorted(coins - trained - excluded)
    end = LAST_DAY + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=DAYS)
    since = start - pd.Timedelta(days=9)
    arms = {"A both": ("both", "both"), "B bin15": ("binance", "both"), "C binance": ("binance", "binance")}
    res = {a: [] for a in arms}
    print(f"  бектест: {len(coins)} монет, {start:%m-%d} – {LAST_DAY:%m-%d}…", flush=True)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for n, b in enumerate(coins, 1):
            raws = {"both": load_raw(conn, [b], INTERVAL_15M, since=since, exchanges=BOTH),
                    "binance": load_raw(conn, [b], INTERVAL_15M, since=since, exchanges=BIN)}
            if raws["binance"]["ohlcv"].empty:
                continue
            f15 = {k: build_from_raw(r, None, bars_per_hour=4, extra=short_features) for k, r in raws.items()}
            h1 = {k: build_from_raw(hourly_from_intraday(r)) for k, r in raws.items()}
            if any(x.empty for x in (*f15.values(), *h1.values())):
                continue
            h = aggregate(raws["binance"], b, "15min")         # ціна угод — та сама для всіх варіантів (Binance)
            h = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
            for arm, (src15, src1) in arms.items():
                mo, thr = models[src15]
                s15 = f15[src15][["base_asset", "ts"]].assign(probability=mo.predict_proba(f15[src15][MODEL_FEATURES_15M])[:, 1], threshold=thr)
                hh = h1[src1]
                s1 = hh[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(hh[m1["features"]])[:, 1], threshold=m1["threshold"])
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
                res[arm] += [{**x, "band": band.get(b, "")} for x in P.simulate(h, sc, K, exit_bars, 3.0, 0.5, True)]
            if n % 40 == 0:
                print(f"    {n}/{len(coins)}", flush=True)

    t_lto = pd.DataFrame(lto).T
    t_bt = pd.DataFrame({k: stats(pd.DataFrame(v)) for k, v in res.items()}).T
    lines = ["# Лише Binance проти Binance + Bybit", "",
             f"Навчання: {len(trained)} тікерів, {int((pumps['pattern_type'] == 'pump').sum())} імпульсів, {len(common)} спільних 15m-барів; "
             f"ознаки без ліквідацій ({len(MODEL_FEATURES_15M)}). OI / L/S Binance дозаповнено з архіву.", "",
             "## 1. Leave-ticker-out (15m, рівень «Кандидат»)", "", md(t_lto), "",
             f"## 2. Бектест: {len(coins)} монет поза навчанням, {start:%Y-%m-%d} – {LAST_DAY:%Y-%m-%d}, частковий вихід, стоп {K} ATR", "",
             md(t_bt), ""]
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {OUT}")


if __name__ == "__main__":
    main()
