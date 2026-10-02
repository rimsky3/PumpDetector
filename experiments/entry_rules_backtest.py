"""
experiments/entry_rules_backtest.py — широкий бектест двох правил входу разом (2026-10-02).

Правила з experiments/book_spot.py, book_filter.py і limit_entry.py, що пройшли перевірку на стабільність у часі:
    фільтр стакану   вхід лише якщо перекіс стакану ±2% ≥ 0 (покупців не менше, ніж продавців; поріг не підбирався)
    ліміт −0.5 ATR   замість купівлі по ринку — лімітна заявка на закриття − 0.5·ATR(15m), дійсна 1 год
Варіанти: як зараз / + фільтр / + ліміт / обидва. Вихід скрізь поточний (risk.PARTIAL_EXIT, 4 бари без Alert, стоп k·ATR).
Перекіс — на момент кожного сигналу (закриття сигнального бару) з архіву Binance bookDepth (кеш data/book_cache/,
бракуючі дні докачуються); сигнал без даних стакану у варіантах з фільтром пропускається.
Монети — 219 поза навчанням (кеші exit_structure.py і style_backtest.py), 30 днів; розбивка за оборотом (5–20 / ≥ 20 / ≥ 50 млн)
і за половинами періоду (стабільність).

    python experiments/entry_rules_backtest.py
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

import book_spot as BS
import limit_entry as L
import train

B15 = pd.Timedelta(minutes=15)
RUN50 = "20261002_0034_current_alert"           # універсум ≥ 50 млн
OUT = os.path.join("reports", "entry_rules_trades.csv")


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def main() -> None:
    argparse.ArgumentParser(description="Широкий бектест: фільтр стакану ±2% і ліміт −0.5 ATR (окремо і разом)").parse_args()
    m15 = joblib.load(train.model_path(train.M15_NAME))
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    alerts = pickle.load(open(L.ALERTS, "rb"))
    bars = pickle.load(open(L.BARS, "rb"))
    band = {}
    for run in L.RUNS:
        d = json.load(open(os.path.join("reports", "backtests", run + ".json"), encoding="utf-8"))
        for u in d["meta"]["universe"]:
            band[u["base"]] = "≥ 20 млн" if d["meta"]["min_turnover"] >= 20e6 else "5–20 млн"
    big = {u["base"] for u in json.load(open(os.path.join("reports", "backtests", RUN50 + ".json"), encoding="utf-8"))["meta"]["universe"]}
    feats = pd.read_csv(os.path.join("reports", "book_spot_features.csv"))
    sym = dict(zip(feats["ticker"], feats["sym"]))
    variants = {"як зараз (ринок)": ("market", 1, 0, 0, False), "+ фільтр стакану": ("market", 1, 0, 0, True),
                "+ ліміт −0.5 ATR": ("atr", 1, 0.5, 4, False), "фільтр + ліміт": ("atr", 1, 0.5, 4, True)}
    rows = []
    coins = sorted(set(alerts) & set(bars))
    for n, b in enumerate(coins, 1):
        t, _ = alerts[b]
        h = bars[b]
        h = h[h.index >= t.index.min() - pd.Timedelta(days=1)]
        sc = t["alert_score"].reindex(h.index)
        al = (sc >= 1).to_numpy()
        starts = [i for i in range(1, len(h)) if al[i] and not al[i - 1]]
        s = sym.get(b, f"{b}USDT")
        days = sorted({d for i in starts for d in ((h.index[i] - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), h.index[i].strftime("%Y-%m-%d"))})
        w = BS.book_wide(s, days) if days else pd.DataFrame()
        imb = {i: BS.book_features(w, h.index[i] + B15).get("imb_2", np.nan) for i in starts}
        for name, (mode, ww, x, valid, filt) in variants.items():
            allow = (lambda i: imb.get(i, np.nan) >= 0) if filt else None
            for r in L.simulate(h, sc, k, exit_bars, mode, ww, x, valid, allow):
                rows.append({"variant": name, "ticker": b, "ts": h.index[r["sig"]], "ret": r["ret"], "reason": r["reason"],
                             "band": band.get(b, ""), "big50": b in big})
        if n % 25 == 0:
            print(f"  {n}/{len(coins)} монет", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    mid = d["ts"].min() + (d["ts"].max() - d["ts"].min()) / 2
    base_n = (d["variant"] == "як зараз (ринок)").sum()
    out = {}
    for name in variants:
        g = d[d["variant"] == name]
        v = g["ret"].sort_values()
        out[name] = {"угод": len(v), "від поточних": f"{len(v) / base_n:.0%}", "у плюсі": f"{(v > 0).mean():.0%}",
                     "медіана": f"{v.median():+.2%}", "EV": f"{v.mean():+.2%}", "PF": round(pf(v), 2),
                     "PF без топ-3": round(pf(v.iloc[:-3]), 2),
                     "5–20 млн": round(pf(g[g["band"] == "5–20 млн"]["ret"]), 2), "≥ 20 млн": round(pf(g[g["band"] == "≥ 20 млн"]["ret"]), 2),
                     "≥ 50 млн": round(pf(g[g["big50"]]["ret"]), 2),
                     "1-а половина": round(pf(g[g["ts"] < mid]["ret"]), 2), "2-а половина": round(pf(g[g["ts"] >= mid]["ret"]), 2),
                     "по стопу": f"{(g['reason'] == 'stop').mean():.0%}"}
    pd.set_option("display.width", 260)
    print(f"\n{len(coins)} монет поза навчанням, вихід — частковий; половини — до / після {mid:%m-%d %H:%M} (PF)\n")
    print(pd.DataFrame(out).T.to_string())
    print(f"\nУгоди: {OUT}")


if __name__ == "__main__":
    main()
