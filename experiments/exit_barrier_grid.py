"""
experiments/exit_barrier_grid.py — чи стійкий вихід «бар'єри» (ціль +TP·ATR / стоп −SL·ATR / максимум H) для входів робочої
моделі (2026-10-02).

experiments/exit_structure.py, варіант E (+3 / −1.5 ATR, 4 год): PF 0.85 проти 0.74 зараз; на монетах з оборотом ≥ 20 млн —
1.06. Перш ніж міняти правило, дві перевірки:
    1) сітка сусідніх параметрів на тих самих 220 монетах бектесту (кеш exit_structure.py): якщо поруч теж краще за поточний
       вихід — це закономірність, а не вдалий збіг одного набору;
    2) журнал скринера (угоди Alert поза навчанням, з 24.09): поточний вихід проти бар'єрів — інший період.

    python experiments/exit_barrier_grid.py
"""

import argparse
import itertools
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import exit_structure as X
import train
from features import INTERVAL_15M, aggregate, load_raw

TPS, SLS, HS = (2.0, 3.0, 4.0), (1.0, 1.5, 2.0), (12, 16, 24)


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def run(data: dict, band: dict, k: float, exit_bars: int, mode: str) -> pd.DataFrame:
    rows = []
    for b, (t, h) in data.items():
        for x in X.simulate(h, t["alert_score"], k, exit_bars, mode, 0):
            rows.append({**x, "band": band.get(b, "")})
    return pd.DataFrame(rows)


def summ(r: pd.DataFrame) -> dict:
    v = r["ret"]
    big = r[r["band"] == "≥ 20 млн"]["ret"]
    w = v.sort_values()
    return {"угод": len(v), "PF": round(pf(v), 2), "PF ≥ 20 млн": round(pf(big), 2), "PF 5–20 млн": round(pf(r[r["band"] == "5–20 млн"]["ret"]), 2),
            "медіана": f"{v.median():+.2%}", "без топ-3": f"{w.iloc[:-3].sum():+.0%}", "по стопу": f"{(r['reason'] == 'stop').mean():.0%}"}


def main() -> None:
    argparse.ArgumentParser(description="Стійкість виходу «бар'єри» для входів робочої моделі (експеримент)").parse_args()
    load_dotenv(".env")
    m15 = joblib.load(train.model_path(train.M15_NAME))
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    data = pickle.load(open(X.CACHE, "rb"))
    band = {}
    for run_id in X.RUNS:
        d = json.load(open(os.path.join("reports", "backtests", run_id + ".json"), encoding="utf-8"))
        for u in d["meta"]["universe"]:
            band[u["base"]] = "≥ 20 млн" if d["meta"]["min_turnover"] >= 20e6 else "5–20 млн"

    # 1) сітка на бектесті
    grid = {"поточний вихід (4 бари без Alert, стоп 2.5 ATR)": summ(run(data, band, k, exit_bars, "atr"))}
    for tp, sl, hb in itertools.product(TPS, SLS, HS):
        X.BAR_TP, X.BAR_SL, X.BAR_H = tp, sl, hb
        grid[f"+{tp:g} / −{sl:g} ATR, {hb / 4:g} год"] = summ(run(data, band, k, exit_bars, "barrier"))
    g = pd.DataFrame(grid).T
    pd.set_option("display.width", 200)
    print(f"1) Бектест, {len(data)} монет поза навчанням, ті самі входи (Alert):\n")
    print(g.to_string())
    br = g.iloc[1:]
    print(f"\n   бар'єри краще за поточний вихід (PF): {(br['PF'] > g.iloc[0]['PF']).sum()} з {len(br)} наборів; "
          f"PF ≥ 20 млн > 1: {(br['PF ≥ 20 млн'] > 1).sum()} з {len(br)}; медіана PF сітки {br['PF'].median():.2f}")

    # 2) журнал скринера: перші оцінки моделі (як у журналі дашборду) на монетах поза навчанням
    import journal as J
    from tiers import TIER_MODEL
    trained = set(m15.get("tickers", []))
    X.BAR_TP, X.BAR_SL, X.BAR_H = 3.0, 1.5, 16
    res = {"поточний": [], "бар'єри +3/−1.5, 4 год": []}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        sc = J.first_scores(conn, TIER_MODEL, m15.get("confirm", {}).get("share", 0.75))
        sc = sc[~sc["base_asset"].isin(trained) & (sc["kind"] != "in-sample")]
        for b, gg in sc.groupby("base_asset"):
            h = aggregate(load_raw(conn, [b], INTERVAL_15M, since=gg["ts"].min() - pd.Timedelta(days=2)), b, "15min")
            if h.empty:
                continue
            s = gg.set_index("ts")["probability"]
            for name, mode in (("поточний", "atr"), ("бар'єри +3/−1.5, 4 год", "barrier")):
                res[name] += [{**x, "band": ""} for x in X.simulate(h, s, k, exit_bars, mode, 0)]
    print(f"\n2) Журнал скринера (монети поза навчанням, з {sc['ts'].min():%m-%d}):\n")
    for name, rows in res.items():
        r = pd.DataFrame(rows)
        v = r["ret"].sort_values()
        print(f"   {name:<26} {len(v):>4} угод, у плюсі {(v > 0).mean():.0%}, медіана {v.median():+.2%}, PF {pf(v):.2f}, "
              f"без топ-3 PF {pf(v.iloc[:-3]):.2f}")


if __name__ == "__main__":
    main()
