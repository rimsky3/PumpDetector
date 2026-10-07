"""
experiments/exit_partial.py — частковий вихід для входів робочої моделі (2026-10-02).

experiments/exit_barrier_grid.py: бар'єри (ціль / стоп / час) кращі за поточний вихід у 27 з 27 наборів (PF 0.81 проти 0.74),
але обрізають великі імпульси (龙虾 +79% → +3 ATR), які й дають заробіток. Компроміс: частину позиції FRAC закрити на цілі
+TP·ATR, решту вести як зараз (вихід — 4 бари без Alert або стоп). Після часткового виходу стоп на решту — той самий або
в беззбиток (BE). Вхід — як у дашборді (Alert), стоп від входу −SL·ATR. Комісія 0.1% на угоду.

Результат угоди = FRAC × (ціль) + (1 − FRAC) × (вихід решти), у частках повної позиції.

    python experiments/exit_partial.py
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
from risk import atr

FEE = 0.001


def simulate(h: pd.DataFrame, score: pd.Series, sl: float, exit_bars: int, tp: float | None, frac: float, be: bool) -> list[dict]:
    a = atr(h).reindex(h.index).to_numpy()
    sc = score.reindex(h.index).to_numpy()
    hi, lo, c = h["high"].to_numpy(), h["low"].to_numpy(), h["price"].to_numpy()
    out, tr, below = [], None, 0
    for i in range(len(h)):
        p = sc[i]
        if np.isnan(p):
            continue
        if tr is None:
            if p >= 1 and not np.isnan(c[i]) and not np.isnan(a[i]):
                tr = {"e": c[i], "stop": c[i] - sl * a[i], "tp": c[i] + tp * a[i] if tp else None, "part": 0.0, "i0": i}
                below = 0
            continue
        e = tr["e"]
        if lo[i] <= tr["stop"]:                                            # стоп (консервативно — раніше за ціль у тому ж барі)
            rest = tr["stop"] / e - 1
            out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - FEE, "reason": "stop",
                        "took": bool(tr["part"]), "bars": i - tr["i0"], "ts": h.index[tr["i0"]]})
            tr = None
            continue
        if tr["tp"] is not None and not tr["part"] and hi[i] >= tr["tp"]:  # частковий вихід на цілі
            tr["part"] = frac * (tr["tp"] / e - 1)
            if be:
                tr["stop"] = max(tr["stop"], e)
        below = below + 1 if p < 1 else 0
        if below >= exit_bars:
            rest = c[i] / e - 1
            out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - FEE, "reason": "model",
                        "took": bool(tr["part"]), "bars": i - tr["i0"], "ts": h.index[tr["i0"]]})
            tr = None
    return out


def summ(rows: list[dict]) -> dict:
    r = pd.DataFrame(rows)
    v = r["ret"].sort_values()
    gl = -v[v <= 0].sum()
    pf = lambda x: round(x[x > 0].sum() / -x[x <= 0].sum(), 2) if (x <= 0).any() else np.nan  # noqa: E731
    return {"угод": len(v), "PF": pf(v), "PF ≥ 20 млн": pf(r[r["band"] == "≥ 20 млн"]["ret"]),
            "PF 5–20 млн": pf(r[r["band"] == "5–20 млн"]["ret"]), "медіана": f"{v.median():+.2%}",
            "без топ-3": f"{v.iloc[:-3].sum():+.0%}", "найкраща": f"{v.iloc[-1]:+.0%}", "по стопу": f"{(r['reason'] == 'stop').mean():.0%}",
            "взято частку": f"{r['took'].mean():.0%}"}


def main() -> None:
    argparse.ArgumentParser(description="Частковий вихід (частина на цілі, решта — за моделлю) для входів робочої моделі").parse_args()
    load_dotenv(".env")
    m15 = joblib.load(train.model_path(train.M15_NAME))
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    data = pickle.load(open(X.CACHE, "rb"))
    band = {}
    for run_id in X.RUNS:
        d = json.load(open(os.path.join("reports", "backtests", run_id + ".json"), encoding="utf-8"))
        for u in d["meta"]["universe"]:
            band[u["base"]] = "≥ 20 млн" if d["meta"]["min_turnover"] >= 20e6 else "5–20 млн"

    def bt(sl, tp, frac, be):
        rows = []
        for b, (t, h) in data.items():
            rows += [{**x, "band": band.get(b, "")} for x in simulate(h, t["alert_score"], sl, exit_bars, tp, frac, be)]
        return rows
    res = {f"поточний (стоп {k:.1f} ATR, без часткового)": summ(bt(k, None, 0, False)),
           "поточний вхід, стоп 2 ATR, без часткового": summ(bt(2.0, None, 0, False))}
    for sl, tp, frac, be in itertools.product((2.0, k), (2.0, 3.0, 4.0), (1 / 3, 0.5), (False, True)):
        res[f"стоп {sl:.1f} · {('½' if frac == 0.5 else '⅓')} на +{tp:g} ATR · {'BE' if be else 'стоп той самий'}"] = summ(bt(sl, tp, frac, be))
    g = pd.DataFrame(res).T
    pd.set_option("display.width", 220)
    print(f"Бектест, {len(data)} монет поза навчанням, входи — Alert робочої моделі:\n")
    print(g.to_string())
    base = g.iloc[0]["PF"]
    part = g.iloc[2:]
    print(f"\nчастковий краще за поточний (PF): {(part['PF'] > base).sum()} з {len(part)}; медіана PF {part['PF'].median():.2f}; "
          f"PF ≥ 20 млн > 1: {(part['PF ≥ 20 млн'] > 1).sum()} з {len(part)}")

    # журнал скринера: найкращий за бектестом набір проти поточного
    best = part["PF"].idxmax()
    sl_b = 2.0 if best.startswith("стоп 2.0") else k
    tp_b = float(best.split("+")[1].split(" ")[0])
    frac_b = 0.5 if "½" in best else 1 / 3
    be_b = "BE" in best
    import journal as J
    from tiers import TIER_MODEL
    trained = set(m15.get("tickers", []))
    rows = {"поточний": [], best: []}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        sc = J.first_scores(conn, TIER_MODEL, m15.get("confirm", {}).get("share", 0.75))
        sc = sc[~sc["base_asset"].isin(trained) & (sc["kind"] != "in-sample")]
        for b, gg in sc.groupby("base_asset"):
            h = aggregate(load_raw(conn, [b], INTERVAL_15M, since=gg["ts"].min() - pd.Timedelta(days=2)), b, "15min")
            if h.empty:
                continue
            s = gg.set_index("ts")["probability"]
            rows["поточний"] += [{**x, "band": ""} for x in simulate(h, s, k, exit_bars, None, 0, False)]
            rows[best] += [{**x, "band": ""} for x in simulate(h, s, sl_b, exit_bars, tp_b, frac_b, be_b)]
    print(f"\nЖурнал скринера (поза навчанням, з {sc['ts'].min():%m-%d}):")
    for name, r in rows.items():
        v = pd.Series([x["ret"] for x in r]).sort_values()
        pf = lambda x: x[x > 0].sum() / -x[x <= 0].sum()  # noqa: E731
        print(f"  {name:<46} {len(v):>3} угод, медіана {v.median():+.2%}, PF {pf(v):.2f}, без топ-3 PF {pf(v.iloc[:-3]):.2f}, "
              f"найкраща {v.iloc[-1]:+.0%}")


if __name__ == "__main__":
    main()
