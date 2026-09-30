"""
experiments/early_phase.py — ознаки ранньої фази пампу: чи може 15m-модель сигналити раніше, ніж ціна вже виросла.

Проблема: у важливості ознак домінує ret_24h — модель частково «бачить, що вже виросло», і сигнал приходить,
коли помітна частина руху позаду. Порівнюємо варіанти 15m-моделі (LightGBM, leave-ticker-out, один поріг з OOF за F1 —
однаково для всіх варіантів, тож порівняння чесне, хоч абсолютні цифри трохи оптимістичні):

    base       — FEATURES_15M, як у роботі
    no_hind    — без ознак «вже виросло»: ret_24h, ret_4h, oi_chg_24h, oi_minus_ret_24h
    early      — base + ознаки ранньої фази (EARLY)
    early_nh   — early без ознак «вже виросло»

Головна метрика — «пройдено руху»: яку частку руху пампу (від ціни на старті до максимуму в інтервалі) ціна вже
пройшла на першому сигналі в пампі (медіана по спійманих пампах). Плюс спіймані пампи, затримка, фальстарти,
хибні епізоди на 30 днів (train.event_metrics, рівень «Кандидат» — лише 15m-модель).

    python experiments/early_phase.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from features import FEATURES_15M, INTERVAL_15M, build_from_raw, in_pump, load_pumps, load_raw, short_features
from fetch import read_tickers_file

HINDSIGHT = ["ret_24h", "ret_4h", "oi_chg_24h", "oi_minus_ret_24h"]
EARLY = ["oi_accel_4h", "oi_up_share_2h", "vol_accel_1h", "oi_vs_ret_1h", "squeeze_4h_24h", "buy_trend_1h"]


def early_features(h: pd.DataFrame, b: int) -> pd.DataFrame:
    """Ознаки ранньої фази — лише дані до бару включно, відносні величини."""
    f = pd.DataFrame(index=h.index)
    lp, oi, vol = np.log(h["price"]), h["oi_log_idx"], h["volume"]
    f["oi_accel_4h"] = oi.diff(b) - oi.diff(4 * b) / 4                     # остання година проти середньої за 4 год
    f["oi_up_share_2h"] = (oi.diff(1) > 0).astype(float).rolling(2 * b).mean()   # стійкість набору позицій
    f["vol_accel_1h"] = vol.rolling(b).sum() / (vol.shift(b).rolling(3 * b).sum() / 3)
    f["oi_vs_ret_1h"] = oi.diff(b) - lp.diff(b)                             # OI набирають, ціна ще стоїть
    r = lp.diff()
    f["squeeze_4h_24h"] = r.rolling(4 * b).std() / r.rolling(24 * b).std()  # стиснення перед пробоєм
    buy, vbv = h["buy_volume"], h["volume_bv"]
    f["buy_trend_1h"] = buy.rolling(b).sum() / vbv.rolling(b).sum() - buy.rolling(24 * b).sum() / vbv.rolling(24 * b).sum()
    return f.replace([np.inf, -np.inf], np.nan)


def moved_share(d: pd.DataFrame, signal: np.ndarray, pumps: pd.DataFrame) -> list[float]:
    """Для кожного спійманого пампу: (ціна на першому сигналі − ціна на старті) / (максимум у пампі − ціна на старті)."""
    out = []
    sig = d.loc[signal, ["base_asset", "ts"]]
    for p in pumps[pumps["pattern_type"] == "pump"].itertuples():
        g = d[d["base_asset"] == p.base_asset]
        if g.empty:
            continue
        inside = in_pump(g["ts"], p.start_ts, p.end_ts, train.BAR_15M).to_numpy()
        s = sig[(sig["base_asset"] == p.base_asset) & in_pump(sig["ts"], p.start_ts, p.end_ts, train.BAR_15M)]
        before = g[g["ts"] + train.BAR_15M <= p.start_ts]
        if s.empty or before.empty or not inside.any():
            continue
        p0, pmax = before["price"].iloc[-1], g.loc[inside, "price"].max()
        ps = g.loc[g["ts"] == s["ts"].min(), "price"].iloc[0]
        if pmax > p0:
            out.append(float(np.clip((ps - p0) / (pmax - p0), 0, 1)))
    return out


def main() -> None:
    argparse.ArgumentParser(description="Ознаки ранньої фази пампу для 15m-моделі (експеримент)").parse_args()
    load_dotenv(".env")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        bases = sorted(reviewed | set(pumps["base_asset"]))
        d = build_from_raw(load_raw(conn, bases, INTERVAL_15M), pumps, bars_per_hour=4,
                           extra=lambda h, b: short_features(h, b).join(early_features(h, b)))
    # вибірка як у train.py: переглянуті тікери — усі бари, інші — лише розмічені інтервали; свіжі нерозмічені — ні
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()].reset_index(drop=True)
    y, groups = d["label"].astype(int), d["base_asset"]
    print(f"{len(d)} барів 15m, {groups.nunique()} тікерів, пампів {int((pumps['pattern_type'] == 'pump').sum())}")

    variants = {
        "base": FEATURES_15M,
        "no_hind": [f for f in FEATURES_15M if f not in HINDSIGHT],
        "early": FEATURES_15M + EARLY,
        "early_nh": [f for f in FEATURES_15M if f not in HINDSIGHT] + EARLY,
    }
    rows = {}
    for name, cols in variants.items():
        print(f"  {name}: {len(cols)} ознак…")
        p = train.oof_proba(train.make_lightgbm, d[cols], y, groups)
        thr = train.best_f1_threshold(y.to_numpy(), p)
        sig = p >= thr
        ev = train.event_metrics(d, sig, pumps, train.BAR_15M)
        ms = moved_share(d, sig, pumps)
        rows[name] = {"ознак": len(cols), "поріг": round(thr, 3), "спіймано": ev["pumps_caught"],
                      "пройдено руху (медіана)": f"{np.median(ms):.0%}" if ms else "—",
                      "пройдено ≤ 25%": f"{np.mean(np.array(ms) <= 0.25):.0%}" if ms else "—",
                      "затримка, хв": round(ev["median_delay_h"] * 60) if pd.notna(ev["median_delay_h"]) else None,
                      "фальстарти": ev["false_starts_triggered"], "хибних / 30 д": round(ev["false_alarms_per_30d"], 1)}
        if name == "early_nh":
            m = train.make_lightgbm().fit(d[cols], y)
            imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
            top = (imp / imp.sum()).head(12)
    print("\n" + pd.DataFrame(rows).T.to_string())
    print("\nВажливість ознак (early_nh, частка сплітів), топ-12:")
    for f, v in top.items():
        print(f"  {f:<20} {v:.1%}" + ("   ← нова" if f in EARLY else ""))


if __name__ == "__main__":
    main()
