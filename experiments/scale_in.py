"""
experiments/scale_in.py — вхід частинами: частина позиції на «Кандидаті», добір на Alert.

Out-of-fold сигнали 15m-контуру (models/oof_15m.parquet; Alert = 15m ≥ 1 і 1h ≥ 0.75 закритої години, як у роботі).
Угода відкривається на першому барі Кандидата (s15 ≥ 1) нового епізоду. Схеми:

    alert_only     усе на першому Alert (як зараз), вихід — Alert згас 4 бари або стоп
    cand_full      усе на Кандидаті, вихід — 15m нижче порогу 4 бари або стоп (без підтвердження)
    scale_f        частка f на Кандидаті; якщо прийшов Alert — добір решти за ціною того бару.
                   Поки Alert не було: вихід, коли Кандидат згас 4 бари; після Alert — коли Alert згас 4 бари.

Стоп один на всю позицію: перша ціна входу − k·ATR(15m), k з бандла (як у роботі). Комісія 0.1% на кожну частину.
Результат угоди — у частках ПОВНОЇ позиції (щоб схеми були порівнянні): 1/3 позиції з −3% = −1%.

    python experiments/scale_in.py
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
from features import INTERVAL_15M, aggregate, in_pump, load_pumps, load_raw
from risk import atr

FEE = 0.001
EXIT_BARS = 4


def simulate(h: pd.DataFrame, s15: pd.Series, alert: pd.Series, k: float, f: float | None) -> list[dict]:
    """f — частка на Кандидаті; None — лише Alert; 1.0 — усе на Кандидаті без доборів."""
    a = atr(h).to_numpy()
    s15, al = s15.reindex(h.index).to_numpy(), alert.reindex(h.index).fillna(False).to_numpy()
    px, lo = h["price"].to_numpy(), h["low"].to_numpy()
    trades, pos, prev_c, prev_a = [], None, False, False
    for i in range(len(px)):
        c_on = bool(s15[i] >= 1) if not np.isnan(s15[i]) else False
        a_on = bool(al[i])
        if pos is None:
            start = (a_on and not prev_a) if f is None else (c_on and not prev_c)
            if start and not np.isnan(a[i]):
                pos = {"t0": h.index[i], "legs": [(px[i], 1.0 if f is None else f)], "stop": px[i] - k * a[i],
                       "alerted": f is None, "below": 0}
                if f is not None and a_on and f < 1:          # Кандидат і Alert на тому ж барі — одразу повна
                    pos["legs"].append((px[i], 1 - f)); pos["alerted"] = True
            prev_c, prev_a = c_on, a_on
            continue
        if lo[i] <= pos["stop"]:
            exit_px, reason = pos["stop"], "stop"
        else:
            if f is not None and f < 1 and not pos["alerted"] and a_on:   # добір на першому Alert
                pos["legs"].append((px[i], 1 - f)); pos["alerted"] = True
            on = a_on if pos["alerted"] and f != 1.0 else c_on
            pos["below"] = 0 if on else pos["below"] + 1
            exit_px, reason = (px[i], "rule") if pos["below"] >= EXIT_BARS else (None, None)
        if exit_px is not None:
            ret = sum(w * (exit_px / e - 1 - FEE) for e, w in pos["legs"])
            trades.append({"t0": pos["t0"], "t1": h.index[i], "ret": ret, "size": sum(w for _, w in pos["legs"]),
                           "alerted": pos["alerted"], "reason": reason})
            pos = None
        prev_c, prev_a = c_on, a_on
    return trades


def main() -> None:
    argparse.ArgumentParser(description="Вхід частинами: Кандидат + добір на Alert (експеримент)").parse_args()
    load_dotenv(".env")
    oof = pd.read_parquet(os.path.join(train.MODELS_DIR, "oof_15m.parquet"))
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    b15 = joblib.load(train.model_path(train.M15_NAME))
    k, share = b15["risk"]["k"], b15.get("confirm", {}).get("share", 0.75)
    oof["s15"] = oof["p15"] / oof["thr15"]
    oof["alert"] = (oof["s15"] >= 1) & (oof["s1h"].fillna(0) >= share)
    bases = sorted(oof["base_asset"].unique())
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        data = {b: aggregate(load_raw(conn, [b], INTERVAL_15M), b, "15min").dropna(subset=["price", "high", "low"]) for b in bases}
    print(f"{len(bases)} тікерів, стоп {k:.2f} ATR, комісія {FEE:.1%} на частину")
    schemes = {"alert_only": None, "cand_full": 1.0, "scale ⅓ + Alert": 1 / 3, "scale ½ + Alert": 0.5}
    rows = {}
    for name, f in schemes.items():
        allt = []
        for b in bases:
            g = oof[oof["base_asset"] == b].set_index("ts")
            for t in simulate(data[b], g["s15"], g["alert"], k, f):
                own = pumps[(pumps["base_asset"] == b) & (pumps["pattern_type"] == "pump")]
                t["in_pump"] = bool(((own["start_ts"] < t["t1"]) & (own["end_ts"] > t["t0"])).any())
                allt.append(t)
        t = pd.DataFrame(allt)
        fs = t[~t["alerted"]] if f not in (None, 1.0) else t.iloc[0:0]
        rows[name] = {"угод": len(t), "в плюсі": f"{(t['ret'] > 0).mean():.0%}", "сер. на угоду": f"{t['ret'].mean():+.2%}",
                      "сума": f"{t['ret'].sum():+.0%}", "сер. виграш": f"{t.loc[t['ret'] > 0, 'ret'].mean():+.1%}",
                      "сер. програш": f"{t.loc[t['ret'] <= 0, 'ret'].mean():+.1%}", "найгірша": f"{t['ret'].min():+.1%}",
                      "по стопу": f"{(t['reason'] == 'stop').mean():.0%}",
                      "в імпульсах, сума": f"{t.loc[t['in_pump'], 'ret'].sum():+.0%}",
                      "поза, сума": f"{t.loc[~t['in_pump'], 'ret'].sum():+.0%}",
                      "без Alert (лише частина)": f"{len(fs)}, сер. {fs['ret'].mean():+.2%}" if len(fs) else "—"}
    print("\n" + pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
