"""
experiments/exit_vwap.py — правило виходу з позиції: «Alert згас 4 бари» проти виходів за VWAP.

Входи — out-of-fold сигнали 15m-контуру (models/oof_15m.parquet: кожен тікер оцінений моделлю, яка його не бачила),
вхід — закриття першого бару нового епізоду Alert (alert_score ≥ 1), стоп — entry − k·ATR(15m) з бандла, як у роботі.
Змінюється ЛИШЕ вихід:

    alert4            як зараз: Alert нижче 1 чотири 15m-бари поспіль
    vwap              закриття нижче VWAP (c барів поспіль, з буфером b·ATR) — Alert ігнорується
    or                що раніше: alert4 або vwap
    and               лише коли обидва: Alert згас 4 бари І ціна нижче VWAP (тримає через «мерехтіння» Alert)

VWAP: anchor — від бару входу; 1h / 4h — ковзний за N годин. Перевірка на закриттях 15m або 5m.
Повторний вхід — лише на НОВОМУ епізоді Alert (перетин знизу вгору), однаково для всіх варіантів.
Комісія 0.1% за угоду (тейкер туди й назад). Відкриті на кінці даних угоди не рахуються.

    python experiments/exit_vwap.py
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
B15, B5 = pd.Timedelta(minutes=15), pd.Timedelta(minutes=5)


def simulate(h15, h5, score, k, rule, anchor, confirm, buf, tf, exit_bars=4):
    """Угоди одного тікера. score — alert_score на 15m-сітці (NaN — оцінки немає)."""
    a15 = atr(h15).to_numpy()
    bars = h5 if tf == "5m" else h15
    mult = 3 if tf == "5m" else 1
    bar = B5 if tf == "5m" else B15
    bp, bh, bl, bv = (bars[c].to_numpy() for c in ("price", "high", "low", "volume"))
    tpv = (bh + bl + bp) / 3 * bv
    roll = {"1h": 4, "4h": 16}.get(anchor)
    if roll:
        w = roll * mult
        vw_roll = (pd.Series(tpv).rolling(w, min_periods=1).sum() / pd.Series(bv).rolling(w, min_periods=1).sum()).to_numpy()
    sc = score.reindex(h15.index).to_numpy()
    # Alert-стан на сітці перевірки: остання 15m-оцінка, відома на закриття бару (as-of)
    s_at = pd.Series(sc, index=h15.index + B15).reindex(bars.index + bar, method="ffill").to_numpy()
    bclose = (bars.index + bar).to_numpy()
    c15 = (h15.index + B15).to_numpy()
    p15 = h15["price"].to_numpy()
    trades, pos, prev_on, j = [], None, False, 0
    for i in range(len(p15)):
        s = sc[i]
        on = bool(s >= 1) if not np.isnan(s) else False
        if pos is None:
            if on and not prev_on and not np.isnan(a15[i]):
                e = p15[i]
                pos = {"entry_ts": h15.index[i], "entry": e, "stop": e - k * a15[i]}
                cpv = cv = 0.0
                below = vbelow = 0
                j = int(np.searchsorted(bclose, c15[i], side="right"))
                if tf == "15m":
                    cpv, cv = tpv[i], bv[i]
            prev_on = on
            continue
        prev_on = on
        while j < len(bp) and bclose[j] <= c15[i]:
            if bl[j] <= pos["stop"]:
                trades.append({**pos, "exit": pos["stop"], "reason": "stop", "exit_ts": bars.index[j]}); pos = None; break
            cpv += tpv[j]; cv += bv[j]
            vw = cpv / cv if anchor == "entry" and cv > 0 else (vw_roll[j] if roll else np.nan)
            vbelow = vbelow + 1 if not np.isnan(vw) and bp[j] < vw - buf * a15[i] else 0
            st = s_at[j]
            below = below + 1 if not (not np.isnan(st) and st >= 1) else 0
            a_hit, v_hit = below >= exit_bars * mult, vbelow >= confirm
            hit = {"alert4": a_hit, "vwap": v_hit, "or": a_hit or v_hit, "and": a_hit and v_hit}[rule]
            j += 1
            if hit:
                trades.append({**pos, "exit": bp[j - 1], "reason": "rule", "exit_ts": bars.index[j - 1]}); pos = None; break
    for t in trades:
        t["ret"] = t["exit"] / t["entry"] - 1 - FEE
        t["hold_h"] = (t["exit_ts"] - t["entry_ts"]) / pd.Timedelta(hours=1)
    return trades


def main() -> None:
    argparse.ArgumentParser(description="Правило виходу: Alert 4 бари проти VWAP (на out-of-fold сигналах)").parse_args()
    load_dotenv(".env")
    oof = pd.read_parquet(os.path.join(train.MODELS_DIR, "oof_15m.parquet"))
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    b15 = joblib.load(train.model_path(train.M15_NAME))
    k, share = b15["risk"]["k"], b15.get("confirm", {}).get("share", 0.75)
    oof["score"] = np.minimum(oof["p15"] / oof["thr15"], oof["s1h"].fillna(0) / share)
    bases = sorted(oof["base_asset"].unique())
    data = {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        for base in bases:
            h15 = aggregate(load_raw(conn, [base], INTERVAL_15M), base, "15min")
            h5 = aggregate(load_raw(conn, [base], "5min"), base, "5min")
            data[base] = (h15.dropna(subset=["price", "high", "low", "volume"]), h5.dropna(subset=["price", "high", "low", "volume"]))
    print(f"{len(bases)} тікерів, k = {k:.2f} ATR, комісія {FEE:.1%} за угоду")

    grid = [("alert4", "-", 0, 0.0, "15m")]
    for rule, anchor, confirm, buf, tf in itertools.product(["vwap", "or", "and"], ["entry", "1h", "4h"], [1, 2, 3], [0.0, 0.5], ["15m", "5m"]):
        grid.append((rule, anchor, confirm, buf, tf))
    rows = []
    for rule, anchor, confirm, buf, tf in grid:
        allt = []
        for base in bases:
            h15, h5 = data[base]
            if h15.empty or (tf == "5m" and h5.empty):
                continue
            sc = oof[oof["base_asset"] == base].set_index("ts")["score"]
            allt += [{**t, "base": base} for t in simulate(h15, h5, sc, k, rule, anchor, confirm, buf, tf)]
        t = pd.DataFrame(allt)
        if t.empty:
            continue
        in_pump = t.apply(lambda r: ((pumps["base_asset"] == r["base"]) & (pumps["pattern_type"] == "pump")
                                     & (pumps["start_ts"] < r["exit_ts"]) & (pumps["end_ts"] > r["entry_ts"])).any(), axis=1)
        rows.append({"правило": rule, "VWAP": anchor, "барів": confirm, "буфер ATR": buf, "ТФ": tf, "угод": len(t),
                     "в плюсі": (t["ret"] > 0).mean(), "очікування": t["ret"].mean(), "сума": t["ret"].sum(),
                     "сер. виграш": t.loc[t["ret"] > 0, "ret"].mean(), "сер. програш": t.loc[t["ret"] <= 0, "ret"].mean(),
                     "по стопу": (t["reason"] == "stop").mean(), "утримання, год": t["hold_h"].median(),
                     "у пампах, сума": t.loc[in_pump, "ret"].sum()})
    r = pd.DataFrame(rows)
    fmt = {"в плюсі": "{:.0%}", "очікування": "{:+.2%}", "сума": "{:+.0%}", "сер. виграш": "{:+.1%}", "сер. програш": "{:+.1%}",
           "по стопу": "{:.0%}", "утримання, год": "{:.1f}", "у пампах, сума": "{:+.0%}"}
    out = r.copy()
    for c, f in fmt.items():
        out[c] = out[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "—")
    base_row = out.iloc[[0]]
    print("\nЯк зараз:\n" + base_row.to_string(index=False))
    print("\nТоп-15 за сумою результатів:\n" + out.iloc[r["сума"].sort_values(ascending=False).index[:15]].to_string(index=False))
    # стабільність: середнє по сусідніх налаштуваннях (усі barів/буфери) для кожної пари правило × VWAP × ТФ
    st = r.iloc[1:].groupby(["правило", "VWAP", "ТФ"])[["сума", "очікування", "угод"]].agg(["mean", "min"])
    print("\nСтабільність (середнє і мінімум по барах підтвердження та буферах):")
    print(st.round(3).sort_values(("сума", "mean"), ascending=False).to_string())
    r.to_csv(os.path.join("reports", "exit_vwap.csv"), index=False)


if __name__ == "__main__":
    main()
