"""
experiments/level_features.py — ознаки рівнів для 15m-моделі (2026-10-02): чи відрізнить модель сплеск на хаях від входу
на відкаті.

experiments/my_entries_profile.py: власник входить на −6% від хаю доби, майже ніколи на новому хаї тижня (4% входів);
модель — на −1% від хаю, у 40% на новому хаї тижня. Моделі бракує «де ціна відносно рівнів». Нові ознаки (LEVELS),
лише дані до бару включно, відносні величини:
    dist_hi_24h      log(ціна / максимум high за 24 год)                    ≤ 0: скільки до хаю доби
    dist_hi_7d       log(ціна / максимум high за 7 днів)                    ≤ 0: скільки до хаю тижня
    range_pos_24h    (ціна − лоу 24 год) / (хай − лоу 24 год)               1 — на хаях доби, 0 — на дні
    above_hi_7d      log(ціна / максимум high попередніх 7 днів без останньої години)  > 0 — пробій тижневого хаю
    dist_hi_7d_atr   (ціна − хай 7 днів) / ATR(14) 15m                       те саме в одиницях волатильності

Порівняння (обидві моделі свіжонавчені однаково — LightGBM, вибірка як у train.py, поріг — найкращий F1 на OOF):
    base     FEATURES_15M (як у роботі)
    levels   FEATURES_15M + LEVELS
1) leave-ticker-out на навчальних тікерах: спіймано, затримка, фальстарти, хибні на 30 днів (рівень «Кандидат»);
2) бектест на монетах бектесту ПОЗА навчанням: рівні з годинним підтвердженням (робоча годинна модель, з власних 15m),
   позиції як у дашборді (Alert, 4 бари, стоп k·ATR) і з частковим виходом (½ на +3 ATR, далі беззбиток —
   experiments/exit_partial.py), комісія 0.1%.

    python experiments/level_features.py
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

import train
from features import (FEATURES, FEATURES_15M, INTERVAL_15M, aggregate, build_from_raw, build_hourly_own, load_pumps,
                      load_raw, short_features)
from fetch import read_tickers_file
from risk import atr, position_state
from tiers import tier_frame

LEVELS = ["dist_hi_24h", "dist_hi_7d", "range_pos_24h", "above_hi_7d", "dist_hi_7d_atr"]
FEE, DAYS = 0.001, 30


def level_features(h: pd.DataFrame, b: int) -> pd.DataFrame:
    f = pd.DataFrame(index=h.index)
    lp, hi, lo = np.log(h["price"]), np.log(h["high"]), np.log(h["low"])
    hi24, lo24, hi7 = hi.rolling(24 * b).max(), lo.rolling(24 * b).min(), hi.rolling(7 * 24 * b, min_periods=5 * 24 * b).max()
    f["dist_hi_24h"] = lp - hi24
    f["dist_hi_7d"] = lp - hi7
    f["range_pos_24h"] = (lp - lo24) / (hi24 - lo24)
    f["above_hi_7d"] = lp - hi.shift(b).rolling(7 * 24 * b, min_periods=5 * 24 * b).max()
    f["dist_hi_7d_atr"] = (h["price"] - np.exp(hi7)) / atr(h)
    return f.replace([np.inf, -np.inf], np.nan)


def extra(h, b):
    return short_features(h, b).join(level_features(h, b))


def _pf(v: pd.Series) -> str:
    gl = -v[v <= 0].sum()
    return f"{v[v > 0].sum() / gl:.2f}" if gl else "∞"


def stats(df: pd.DataFrame) -> dict:
    v = df["ret"].sort_values()
    if v.empty:
        return {"угод": 0}
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}", "без топ-3": f"{v.iloc[:-3].sum():+.0%}",
            "PF": _pf(v), "PF без топ-3": _pf(v.iloc[:-3]), "PF ≥ 20 млн": _pf(df[df["band"] == "≥ 20 млн"]["ret"]),
            "PF 5–20 млн": _pf(df[df["band"] == "5–20 млн"]["ret"]), "по стопу": f"{df['stop'].mean():.0%}"}


def main() -> None:
    argparse.ArgumentParser(description="Ознаки рівнів (відстань до хаїв, положення в діапазоні) для 15m-моделі").parse_args()
    load_dotenv(".env")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    k, exit_bars, share = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        bases = sorted(reviewed | set(pumps["base_asset"]))
        d = build_from_raw(load_raw(conn, bases, INTERVAL_15M), pumps, bars_per_hour=4, extra=extra)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()].reset_index(drop=True)
    y, groups = d["label"].astype(int), d["base_asset"]
    trained = set(groups)
    print(f"навчання: {len(d)} барів 15m, {groups.nunique()} тікерів, імпульсів {int((pumps['pattern_type'] == 'pump').sum())}")

    variants = {"base": FEATURES_15M, "levels": FEATURES_15M + LEVELS}
    oof_rows, models = {}, {}
    for name, cols in variants.items():
        print(f"  {name}: {len(cols)} ознак, leave-ticker-out…", flush=True)
        p = train.oof_proba(train.make_lightgbm, d[cols], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        ev = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        oof_rows[name] = {"поріг": round(thr, 3), "спіймано": ev["pumps_caught"],
                          "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                          "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
        models[name] = (train.make_lightgbm().fit(d[cols], y), thr, cols)
    imp = pd.Series(models["levels"][0].booster_.feature_importance("gain"), index=variants["levels"])
    imp = (imp / imp.sum()).sort_values(ascending=False)

    # бектест: монети прогонів поза навчанням
    coins, band = set(), {}
    for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
        meta = json.load(open(path, encoding="utf-8"))["meta"]
        for u in meta["universe"]:
            coins.add(u["base"])
            band[u["base"]] = "≥ 20 млн" if meta["min_turnover"] >= 20e6 else "5–20 млн"
    coins = sorted(coins - trained)
    start = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(days=DAYS)
    since = start - pd.Timedelta(days=9)
    import exit_partial as P
    res = {f"{name} · {e}": [] for name in variants for e in ("поточний вихід", "частковий вихід")}
    print(f"бектест: {len(coins)} монет поза навчанням, {DAYS} днів…", flush=True)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for n, b in enumerate(coins, 1):
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
                t = t[t.index >= start]
                sc = t["alert_score"]
                for x in P.simulate(h, sc, k, exit_bars, None, 0, False):          # = risk.position_state
                    res[f"{name} · поточний вихід"].append({"ticker": b, "ret": x["ret"], "stop": x["reason"] == "stop", "band": band.get(b, "")})
                for x in P.simulate(h, sc, k, exit_bars, 3.0, 0.5, True):         # ½ на +3 ATR, далі беззбиток
                    res[f"{name} · частковий вихід"].append({"ticker": b, "ret": x["ret"], "stop": x["reason"] == "stop", "band": band.get(b, "")})
            if n % 25 == 0:
                print(f"  {n}/{len(coins)}", flush=True)
    bt = {}
    for name, rows in res.items():
        df = pd.DataFrame(rows)
        bt[name] = stats(df) if len(df) else {"угод": 0}
    pd.set_option("display.width", 220)
    print("\n1) Leave-ticker-out на навчальних тікерах (рівень «Кандидат», 15m):")
    print(pd.DataFrame(oof_rows).T.to_string())
    print(f"\n2) Бектест на {len(coins)} монетах поза навчанням (Alert = 15m + 1h, позиції як у дашборді):")
    print(pd.DataFrame(bt).T.to_string())
    print("\nВажливість ознак (levels, частка gain), топ-12:")
    for f, v in imp.head(12).items():
        print(f"  {f:<18} {v:.1%}" + ("   ← нова" if f in LEVELS else ""))


if __name__ == "__main__":
    main()
