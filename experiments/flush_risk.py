"""
experiments/flush_risk.py — чи передбачає «боковик біля максимуму + OI на максимумі» злив лонгів (як NOM 2026-09-30).

Гіпотеза: імпульс закінчився (ціна стоїть, нового максимуму немає), а OI не падає — лонги не закриваються, пастка;
злив стопів лонгів імовірніший, ніж зазвичай після імпульсу.

Всесвіт: 15m-бари протягом 12 год після будь-якого бару з Alert (out-of-fold, models/oof_15m.parquet).
Злив: протягом наступних 4 год low ≤ закриття − 3·ATR(15m).
Умови на барі t (лише минуле):
    stall      1–24 год без нового 24-годинного максимуму і ціна не далі 5% від нього (боковик біля верху)
    oi_top     OI у монетах ≥ 97% свого максимуму за 24 год (позиції не закриваються)
    fading     15m-оцінка падає 3 бари поспіль
    longs      частка лонгів (акаунти, L/S) ≥ 60%
    funding+   funding > 0
Сигнали проріджені: у межах тікера — не частіше разу на 4 год (перший бар умови), щоб один боковик не рахувався 16 разів.

    python experiments/flush_risk.py
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
from features import INTERVAL_15M, aggregate, load_raw
from risk import atr

HORIZON = 16          # 4 год
FLUSH_ATR = 3.0


def bars_for(h: pd.DataFrame, score: pd.Series, s15: pd.Series) -> pd.DataFrame:
    a = atr(h)
    d = pd.DataFrame(index=h.index)
    d["price"], d["atr"] = h["price"], a
    on = (score.reindex(h.index) >= 1).fillna(False)
    last_alert = pd.Series(np.where(on, np.arange(len(h)), np.nan), index=h.index).ffill()
    d["after"] = (np.arange(len(h)) - last_alert <= 48) & last_alert.notna()                 # ≤ 12 год після Alert
    hi24 = h["high"].rolling(96, min_periods=8).max()
    is_new = h["high"] >= hi24
    since_hi = pd.Series(np.arange(len(h)), index=h.index) - pd.Series(np.where(is_new, np.arange(len(h)), np.nan), index=h.index).ffill()
    d["stall"] = (since_hi >= 4) & (since_hi <= 96) & (h["price"] >= 0.95 * hi24)
    coins = h["oi_coins_c"] if "oi_coins_c" in h else h["oi"]
    d["oi_top"] = coins >= 0.97 * coins.rolling(96, min_periods=8).max()
    s = s15.reindex(h.index)
    d["fading"] = (s < s.shift(1)) & (s.shift(1) < s.shift(2)) & (s.shift(2) < s.shift(3))
    d["longs"] = h["ls_long_pct"] >= 60 if "ls_long_pct" in h else False
    d["funding_pos"] = h["funding"] > 0 if "funding" in h else False
    lo_fwd = h["low"][::-1].rolling(HORIZON, min_periods=HORIZON).min()[::-1].shift(-1)
    d["flush"] = lo_fwd <= h["price"] - FLUSH_ATR * a
    d["r4"] = h["price"].shift(-HORIZON) / h["price"] - 1
    d["oi4"] = coins.shift(-HORIZON) / coins - 1
    return d[d["after"] & d["flush"].notna() & lo_fwd.notna()]


def thin(mask: pd.Series) -> pd.Series:
    """Перший бар умови, далі не частіше разу на HORIZON барів."""
    out, last = np.zeros(len(mask), dtype=bool), -10 ** 9
    for i, v in enumerate(mask.to_numpy()):
        if v and i - last >= HORIZON:
            out[i], last = True, i
    return pd.Series(out, index=mask.index)


def main() -> None:
    argparse.ArgumentParser(description="Злив після боковика: боковик + OI на максимумі (експеримент)").parse_args()
    load_dotenv(".env")
    oof = pd.read_parquet(os.path.join(train.MODELS_DIR, "oof_15m.parquet"))
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    b15 = joblib.load(train.model_path(train.M15_NAME))
    share = b15.get("confirm", {}).get("share", 0.75)
    oof["s15"] = oof["p15"] / oof["thr15"]
    oof["score"] = np.minimum(oof["s15"], oof["s1h"].fillna(0) / share)
    parts = []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for b in sorted(oof["base_asset"].unique()):
            h = aggregate(load_raw(conn, [b], INTERVAL_15M), b, "15min").dropna(subset=["price", "high", "low"])
            g = oof[oof["base_asset"] == b].set_index("ts")
            parts.append(bars_for(h, g["score"], g["s15"]).assign(base=b))
    d = pd.concat(parts)
    conds = {
        "усі бари після імпульсу (база)": pd.Series(True, index=d.index),
        "боковик біля максимуму": d["stall"],
        "боковик + OI на максимумі": d["stall"] & d["oi_top"],
        "боковик + OI на максимумі + P слабшає": d["stall"] & d["oi_top"] & d["fading"],
        "боковик + OI на максимумі + лонгів ≥ 60%": d["stall"] & d["oi_top"] & d["longs"],
        "боковик + OI на максимумі + funding > 0": d["stall"] & d["oi_top"] & d["funding_pos"],
        "боковик, OI вже НЕ на максимумі": d["stall"] & ~d["oi_top"],
    }
    base = d["flush"].mean()
    rows = {}
    for name, m in conds.items():
        m = m.fillna(False).astype(bool)
        sig = pd.concat([thin(m[d["base"] == b]) for b in d["base"].unique()]).reindex(d.index).fillna(False).astype(bool)
        x = d[sig]
        rows[name] = {"ситуацій": len(x), "тікерів": x["base"].nunique(), "злив за 4 год": f"{x['flush'].mean():.0%}",
                      "у скільки разів частіше": f"×{x['flush'].mean() / base:.2f}" if base else "—",
                      "ціна за 4 год (мед.)": f"{x['r4'].median():+.1%}", "OI за 4 год (мед.)": f"{x['oi4'].median():+.1%}"}
    print(f"Злив = за 4 год ціна на {FLUSH_ATR:g}+ ATR(15m) нижче; база — бари ≤ 12 год після Alert ({len(d)} барів, 15 тікерів)\n")
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
