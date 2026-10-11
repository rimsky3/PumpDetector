"""
experiments/alert_filters.py — ФІЛЬТРИ ALERT БЕЗ ПЕРЕНАВЧАННЯ (2026-10-11; власник: «як збільшити PF — фільтрувати Алерти? торгувати
лише монети з власним рухом; не торгувати, коли росте весь ринок»; «так»).
Угоди — бектест base з experiments/sample_weights.py (робочі ознаки, розмітка власника, seeds 42 / 43 / 44, 6 міс., чесний список,
частковий вихід). На момент входу рахується контекст; фільтр прибирає угоди (наближення: шлях угод не перераховується — пропущений
вхід міг би відкрити іншу угоду трохи пізніше).
Фільтри (зафіксовано ДО запуску):
    idio_med     власний рух за 24 год (idio_24h) на вході ≥ медіани по всіх входах
    market_cool  ринок не «гарячий»: ≤ 60% монет кешу архіву в плюсі за 7 днів (повтор market_regime на поточних ознаках)
    conc_N       за останні 60 хв входів на ІНШИХ монетах < N (N = 3 / 5 / 10): спалах Alert по ринку = одна корельована ставка
    noreent_H    на цій монеті не було виходу по стопу за останні H год (H = 12 / 24)
КРИТЕРІЙ (до запуску): середнє за seeds — PF на ≥ 20 млн вищий за «без фільтра» щонайменше на 0.03; краще на обох половинах монет;
краще у ≥ 4 з 6 місяців (≥ 20 млн); лишається ≥ 50% угод.
→ reports/alert_filters.md

    python experiments/alert_filters.py
"""

import argparse
import glob
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import idio_features as I

SRC = os.path.join("reports", "sample_weights_trades.csv")
OUT = os.path.join("reports", "alert_filters.md")
BIG = "≥ 20 млн"
BAR = pd.Timedelta(minutes=15)


def breadth_7d() -> pd.Series:
    """Частка монет кешу архіву з додатною 7-денною дохідністю на кожному 15m-барі (≥ 50 монет з даними)."""
    r = {}
    for p in glob.glob(os.path.join(I.DIR, "*.pkl")):
        h = pickle.load(open(p, "rb"))["h"]
        c = h["price"].where(h["price"] > 0)
        r[os.path.basename(p)[:-4]] = (c / c.shift(7 * 96) - 1)
    w = pd.DataFrame(r)
    n = w.notna().sum(axis=1)
    return ((w > 0).sum(axis=1) / n).where(n >= 50)


def idio_at(tr: pd.DataFrame, idx: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=tr.index)
    for c, g in tr.groupby("ticker"):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        nf = I.new_features(x["h"], idx)["idio_24h"]
        out[g.index] = nf.reindex(pd.DatetimeIndex(g["ts"])).to_numpy()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Фільтри Alert без перенавчання: власний рух, гарячий ринок, спалах Alert по ринку, повторний вхід після стопу").parse_args()
    tr = pd.read_csv(SRC, parse_dates=["ts"])
    tr = tr[tr["variant"] == "base"].reset_index(drop=True)
    tr["exit_ts"] = tr["ts"] + tr["bars"] * BAR
    print(f"угод base: {len(tr)} ({tr['seed'].nunique()} seeds); рахую контекст…", flush=True)
    idx = I.market_index()
    br = breadth_7d()
    tr["breadth"] = br.reindex(pd.DatetimeIndex(tr["ts"])).to_numpy()
    tr["idio"] = idio_at(tr, idx)
    conc, stop_h = np.zeros(len(tr)), np.full(len(tr), np.inf)
    for sd, g in tr.groupby("seed"):
        g = g.sort_values("ts")
        t = g["ts"].to_numpy()
        for i, (k, r) in enumerate(g.iterrows()):
            lo = np.searchsorted(t, r["ts"] - pd.Timedelta(minutes=60), side="left")
            win = g.iloc[lo:i]
            conc[k] = (win["ticker"] != r["ticker"]).sum()
        for c, gc in g.groupby("ticker"):
            gc = gc.sort_values("ts")
            last = None
            for k, r in gc.iterrows():
                if last is not None:
                    stop_h[k] = (r["ts"] - last) / pd.Timedelta(hours=1)
                if r["reason"] == "stop":
                    last = r["exit_ts"]
    tr["conc"], tr["since_stop_h"] = conc, stop_h
    med = tr["idio"].median()
    F = {"без фільтра": np.ones(len(tr), bool),
         "idio_med": (tr["idio"] >= med).to_numpy(),
         "market_cool": (tr["breadth"] <= 0.60).to_numpy() | tr["breadth"].isna().to_numpy(),
         **{f"conc_{n}": (tr["conc"] < n).to_numpy() for n in (3, 5, 10)},
         **{f"noreent_{h}": (tr["since_stop_h"] > h).to_numpy() for h in (12, 24)}}
    pf = I.pf
    seeds = sorted(tr["seed"].unique())
    months = sorted(tr["month"].unique())
    res = {}
    for name, m in F.items():
        rows = []
        for sd in seeds:
            g = tr[m & (tr["seed"] == sd).to_numpy()]
            b = g[g["band"] == BIG]
            rows.append({"n": len(g), "pf": pf(g["ret"]), "pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]),
                         "B": pf(b[b["half"] == "B"]["ret"]), **{f"m{mth}": pf(b[b["month"] == mth]["ret"]) for mth in months}})
        res[name] = pd.DataFrame(rows).mean()
    b0 = res["без фільтра"]
    L = ["# Фільтри Alert без перенавчання", "",
         f"Угоди base з sample_weights.py (seeds {', '.join(map(str, seeds))}, 6 міс.). Медіана власного руху на вході {med:+.3f}. "
         "Значення — середнє за seeds. Наближення: шлях угод не перераховується.", "",
         "**Критерій (до запуску):** ≥ 20 млн PF ≥ без фільтра + 0.03; краще на обох половинах; ≥ 4 з 6 місяців (≥ 20 млн); ≥ 50% угод.", "",
         "| фільтр | угод | частка | PF | ≥ 20 млн | різниця ≥ 20 млн | A ≥ 20 млн | B ≥ 20 млн | місяців краще (≥ 20 млн) | критерій |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        better = sum(r[f"m{mth}"] > b0[f"m{mth}"] + 1e-9 for mth in months)
        ok = (r["pf20"] - b0["pf20"] >= 0.03 and r["A"] > b0["A"] and r["B"] > b0["B"] and better >= 4 and r["n"] >= 0.5 * b0["n"])
        L.append(f"| {name} | {r['n']:.0f} | {r['n'] / b0['n']:.0%} | {r['pf']:.3f} | {r['pf20']:.3f} | {r['pf20'] - b0['pf20']:+.3f} | "
                 f"{r['A']:.3f} | {r['B']:.3f} | {'—' if name == 'без фільтра' else f'{better} з {len(months)}'} | "
                 f"{'—' if name == 'без фільтра' else ('**ВИКОНАНО**' if ok else 'ні')} |")
    L += ["", "## Розподіл контексту на вході", "",
          f"- частка входів у «гарячому» ринку (> 60% монет у плюсі за 7 дн): {(tr['breadth'] > 0.6).mean():.0%}",
          f"- входів за попередню годину на інших монетах: медіана {tr['conc'].median():.0f}, 90-й перцентиль {tr['conc'].quantile(0.9):.0f}",
          f"- входів протягом 24 год після стопу на тій самій монеті: {(tr['since_stop_h'] <= 24).mean():.0%}"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
