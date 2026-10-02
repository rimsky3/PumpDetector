"""
experiments/oi_coins_15m.py — OI без ціни (у монетах) у 15m-моделі (2026-10-02).

experiments/feature_corr.py: зміни OI в USD майже дублюють зміни ціни (24h +0.83, 1h +0.89, бар +0.91), бо OI_usd =
монети × ціна; топ-ознака моделі oi_chg_24h ≈ «скільки вже виросла ціна». Монетний OI показує, чи справді відкривають
позиції. На старій годинній розмітці (experiments/oi_coins.py, 29.09) монетний OI був гіршим — тут 15m, нова розмітка,
бектест на нових монетах.

Варіанти 15m-моделі (годинне підтвердження — робоча 1h-модель, без змін; стоп 2.48 ATR — як у робочої):
    base    FEATURES_15M (OI в USD)
    coins   OI-ознаки замінено монетними: oi_chg_1h/4h/24h, oi_z_24h/7d → oi_coin_*; oi_chg_bar/2bar, oi_accel_1h →
            монетні; oi_minus_ret_24h прибрано (монетна зміна вже без ціни)
    both    base + монетні (модель вибирає сама)
Перевірка: leave-ticker-out (рівень «Кандидат») і бектест на монетах бектесту поза навчанням — поточний і частковий вихід.

    python experiments/oi_coins_15m.py
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
from features import FEATURES, FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, build_hourly_own, load_pumps, load_raw, short_features
from fetch import read_tickers_file
from tiers import tier_frame

DAYS, K = 30, 2.48
COIN_SHORT = ["oi_coin_chg_bar", "oi_coin_chg_2bar", "oi_coin_accel_1h"]
SWAP = {"oi_chg_1h": "oi_coin_chg_1h", "oi_chg_4h": "oi_coin_chg_4h", "oi_chg_24h": "oi_coin_chg_24h", "oi_z_24h": "oi_coin_z_24h",
        "oi_z_7d": "oi_coin_z_7d", "oi_chg_bar": "oi_coin_chg_bar", "oi_chg_2bar": "oi_coin_chg_2bar", "oi_accel_1h": "oi_coin_accel_1h"}
VARIANTS = {
    "base": FEATURES_15M,
    "coins": [SWAP.get(f, f) for f in FEATURES_15M if f != "oi_minus_ret_24h"],
    "both": FEATURES_15M + list(SWAP.values()),
}


def extra(h: pd.DataFrame, b: int) -> pd.DataFrame:
    f = short_features(h, b)
    oc = h["oi_coin_log_idx"]
    f["oi_coin_chg_bar"], f["oi_coin_chg_2bar"] = oc.diff(1), oc.diff(2)
    f["oi_coin_accel_1h"] = oc.diff(b) - oc.diff(b).shift(b)
    return f.replace([np.inf, -np.inf], np.nan)


def pf(v: pd.Series) -> str:
    gl = -v[v <= 0].sum()
    return f"{v[v > 0].sum() / gl:.2f}" if gl else "∞"


def stats(df: pd.DataFrame) -> dict:
    v = df["ret"].sort_values()
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "PF": pf(v), "PF без топ-3": pf(v.iloc[:-3]),
            "PF ≥ 20 млн": pf(df[df["band"] == "≥ 20 млн"]["ret"]), "PF 5–20 млн": pf(df[df["band"] == "5–20 млн"]["ret"])}


def main() -> None:
    argparse.ArgumentParser(description="OI у монетах (без ціни) у 15m-моделі: LTO і бектест на нових монетах").parse_args()
    load_dotenv(".env")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        bases = sorted(reviewed | set(pumps["base_asset"]))
        d = build_from_raw(load_raw(conn, bases, INTERVAL_15M), pumps, bars_per_hour=4, extra=extra)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()].reset_index(drop=True)
    y, groups = d["label"].astype(int), d["base_asset"]
    trained = set(groups)
    lto, models, imps = {}, {}, {}
    for name, cols in VARIANTS.items():
        print(f"  {name}: {len(cols)} ознак, leave-ticker-out…", flush=True)
        p = train.oof_proba(train.make_lightgbm, d[cols], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        ev = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        lto[name] = {"спіймано": ev["pumps_caught"], "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                     "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
        model = train.make_lightgbm().fit(d[cols], y)
        models[name] = (model, thr, cols)
        imp = pd.Series(model.booster_.feature_importance("gain"), index=cols)
        imps[name] = (imp / imp.sum()).sort_values(ascending=False)

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
    print(f"  бектест: {len(coins)} монет поза навчанням…", flush=True)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for b in coins:
            raw = load_raw(conn, [b], INTERVAL_15M, since=since)
            f15 = build_from_raw(raw, None, bars_per_hour=4, extra=extra)
            d1 = build_hourly_own(conn, [b], since=since)
            if f15.empty or d1.empty:
                continue
            s1 = d1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(d1[FEATURES])[:, 1], threshold=m1["threshold"])
            h = aggregate(raw, b, "15min")
            h = h[h.index >= start - pd.Timedelta(days=1)]
            for name, (model, thr, cols) in models.items():
                s15 = f15[["base_asset", "ts"]].assign(probability=model.predict_proba(f15[cols])[:, 1], threshold=thr)
                t = tier_frame(s15, s1, share).set_index("ts").sort_index()
                sc = t.loc[t.index >= start, "alert_score"]
                res[f"{name} · поточний вихід"] += [{**x, "band": band.get(b, "")} for x in P.simulate(h, sc, K, exit_bars, None, 0, False)]
                res[f"{name} · частковий вихід"] += [{**x, "band": band.get(b, "")} for x in P.simulate(h, sc, K, exit_bars, 3.0, 0.5, True)]
    pd.set_option("display.width", 220)
    print("\n1) Leave-ticker-out на навчальних тікерах (рівень «Кандидат»):")
    print(pd.DataFrame(lto).T.to_string())
    print(f"\n2) Бектест на {len(coins)} монетах поза навчанням (Alert = 15m + робоча 1h, стоп {K} ATR):")
    print(pd.DataFrame({k: stats(pd.DataFrame(v)) for k, v in res.items()}).T.to_string())
    for name in ("coins", "both"):
        print(f"\nВажливість ознак ({name}, gain), топ-8:")
        for f, v in imps[name].head(8).items():
            print(f"  {f:<18} {v:.1%}" + ("   ← монетний OI" if "coin" in f else ""))


if __name__ == "__main__":
    main()
