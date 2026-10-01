"""
experiments/exit_giveback.py — вихід з позиції за логікою розмітки: «ціна віддала частку руху від максимуму» проти
поточного «Alert згас 4 бари».

Правило розмітки (2026-10-01): кінець імпульсу — закриття нижче «пік − ⅓ руху» (і 3 год без нового максимуму) або OI −10%.
Наживо майбутнього не видно, тож реальна версія без погляду вперед:
    рух = максимум high з моменту входу − ціна входу; вихід, коли закриття 15m < максимум − frac·рух
    (лише після того, як рух ≥ min_move ATR — інакше будь-яке коливання на старті = вихід), або OI ≤ (1−0.1)·максимум
    OI з моменту входу (якщо oi=True), або стоп.

Входи — out-of-fold Alert 15m-контуру (models/oof_15m.parquet), стоп k·ATR з бандла, комісія 0.1%, як у exit_vwap.py.
Варіанти: alert4 (як зараз); give — лише віддача; or — що раніше (give або alert4); and — лише коли обидва
(Alert згас 4 бари І ціна вже віддала частку); and_small — як and, але поки рух < min_move ATR — вихід як зараз. frac ∈ {¼, ⅓, ½}, min_move ∈ {1, 2} ATR, з OI −10% і без.

    python experiments/exit_giveback.py
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from features import INTERVAL_15M, aggregate, load_pumps, load_raw
from risk import atr

FEE = 0.001


def simulate(h, score, k, rule, frac, min_move, use_oi, exit_bars=4):
    a = atr(h).to_numpy()
    sc = score.reindex(h.index).to_numpy()
    px, hi, lo, oi = (h[c].to_numpy() for c in ("price", "high", "low", "oi"))
    trades, pos, prev_on = [], None, False
    for i in range(len(px)):
        on = bool(sc[i] >= 1) if not np.isnan(sc[i]) else False
        if pos is None:
            if on and not prev_on and not np.isnan(a[i]):
                pos = {"i0": i, "e": px[i], "stop": px[i] - k * a[i], "atr": a[i], "pk": px[i],
                       "oimax": oi[i] if not np.isnan(oi[i]) else 0.0, "below": 0}
            prev_on = on
            continue
        prev_on = on
        if lo[i] <= pos["stop"]:
            trades.append((pos, pos["stop"], "stop", i)); pos = None; continue
        pos["pk"] = max(pos["pk"], hi[i])
        if not np.isnan(oi[i]):
            pos["oimax"] = max(pos["oimax"], oi[i])
        pos["below"] = 0 if on else pos["below"] + 1
        a_hit = pos["below"] >= exit_bars
        mv = pos["pk"] - pos["e"]
        g_hit = mv >= min_move * pos["atr"] and px[i] < pos["pk"] - frac * mv
        if use_oi and not np.isnan(oi[i]) and pos["oimax"] and oi[i] <= 0.9 * pos["oimax"]:
            g_hit = True
        big = mv >= min_move * pos["atr"]
        hit = {"alert4": a_hit, "give": g_hit, "or": a_hit or g_hit, "and": a_hit and g_hit,
               # «and_small»: поки рух малий (< min_move ATR) — як зараз; коли рух великий — тримати, доки ціна не віддасть частку
               "and_small": a_hit and (g_hit or not big)}[rule]
        if hit:
            trades.append((pos, px[i], "rule", i)); pos = None
    out = []
    for p, x, why, i in trades:
        out.append({"ret": x / p["e"] - 1 - FEE, "reason": why, "hold_h": (i - p["i0"]) / 4,
                    "t0": h.index[p["i0"]], "t1": h.index[i],
                    "captured": (x - p["e"]) / (p["pk"] - p["e"]) if p["pk"] > p["e"] else np.nan})
    return out


def main() -> None:
    argparse.ArgumentParser(description="Вихід «ціна віддала частку руху» проти «Alert згас 4 бари» (експеримент)").parse_args()
    load_dotenv(".env")
    oof = pd.read_parquet(os.path.join(train.MODELS_DIR, "oof_15m.parquet"))
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    b15 = joblib.load(train.model_path(train.M15_NAME))
    k, share = b15["risk"]["k"], b15.get("confirm", {}).get("share", 0.75)
    oof["score"] = np.minimum(oof["p15"] / oof["thr15"], oof["s1h"].fillna(0) / share)
    bases = sorted(oof["base_asset"].unique())
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        data = {b: aggregate(load_raw(conn, [b], INTERVAL_15M), b, "15min").dropna(subset=["price", "high", "low"]) for b in bases}
    grid = [("alert4", 0, 0, False)] + [(r, f, m, o) for r, f, m, o in
                                        itertools.product(["give", "or", "and", "and_small"], [0.25, 1 / 3, 0.5], [1, 2], [False, True])]
    rows = []
    for rule, frac, mm, use_oi in grid:
        allt = []
        for b in bases:
            sc = oof[oof["base_asset"] == b].set_index("ts")["score"]
            for t in simulate(data[b], sc, k, rule, frac, mm, use_oi):
                own = pumps[(pumps["base_asset"] == b) & (pumps["pattern_type"] == "pump")]
                t["in_pump"] = bool(((own["start_ts"] < t["t1"]) & (own["end_ts"] > t["t0"])).any())
                allt.append(t)
        t = pd.DataFrame(allt)
        rows.append({"правило": rule, "частка": "—" if rule == "alert4" else f"{frac:.2f}", "мін. рух ATR": mm if rule != "alert4" else "—",
                     "OI −10%": ("так" if use_oi else "ні") if rule != "alert4" else "—", "угод": len(t),
                     "в плюсі": (t["ret"] > 0).mean(), "на угоду": t["ret"].mean(), "сума": t["ret"].sum(),
                     "сер. виграш": t.loc[t["ret"] > 0, "ret"].mean(), "сер. програш": t.loc[t["ret"] <= 0, "ret"].mean(),
                     "по стопу": (t["reason"] == "stop").mean(), "утримання, год": t["hold_h"].median(),
                     "забрано руху": t["captured"].median(), "в імпульсах": t.loc[t["in_pump"], "ret"].sum()})
    r = pd.DataFrame(rows)
    fmt = {"в плюсі": "{:.0%}", "на угоду": "{:+.2%}", "сума": "{:+.0%}", "сер. виграш": "{:+.1%}", "сер. програш": "{:+.1%}",
           "по стопу": "{:.0%}", "утримання, год": "{:.1f}", "забрано руху": "{:.0%}", "в імпульсах": "{:+.0%}"}
    out = r.copy()
    for c, f in fmt.items():
        out[c] = out[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "—")
    print("Як зараз:\n" + out.iloc[[0]].to_string(index=False))
    print("\nТоп-12 за сумою:\n" + out.iloc[r["сума"].sort_values(ascending=False).index[:12]].to_string(index=False))
    st = r.iloc[1:].groupby(["правило", "частка"])["сума"].agg(["mean", "min", "max"]).round(2)
    print("\nСтабільність суми по мін. руху та OI (середнє / мін / макс):\n" + st.sort_values("mean", ascending=False).to_string())
    r.to_csv(os.path.join("reports", "exit_giveback.csv"), index=False)


if __name__ == "__main__":
    main()
