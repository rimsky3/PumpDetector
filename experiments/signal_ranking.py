"""
experiments/signal_ranking.py — ВИБІР СЕРЕД ОДНОЧАСНИХ СИГНАЛІВ (2026-10-09, власник: продовження ліміту одночасних позицій
з market_filter.py — там «хто перший», а якщо брати найкращі?). 6 міс. архіву Binance, чесний список, робоча модель (15m +
годинна ≥ 0.75), частковий вихід, стоп I.K.
Ознаки входу (зафіксовано ДО запуску): s15 = P15 / поріг, alert_score, idio_24h (рух понад ринок), beta і r2 до ALT-EW
(менше — більш «власний» рух), open_n — скільки позицій моделі вже відкрито в момент входу (натовп).
1) Чи сортують ознаки угоди взагалі: PF за квінтилями кожної ознаки (усі і ≥ 20 млн).
2) Ліміт N = 3 / 5 одночасних позицій (≥ 20 млн): коли вільних місць менше, ніж сигналів у тому самому барі, — пріоритет:
   first (довільний, як «хто перший»), s15, idio_24h, low_beta, low_r2, random. Плюс «без ліміту».
Стабільність — дві половини (квіт–черв / лип–вер).
→ reports/signal_ranking.md, reports/signal_ranking_trades.csv

    python experiments/signal_ranking.py
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import exit_partial as P
import idio_features as I
import train
from fetch import read_tickers_file
from tiers import tier_frame

BAR = pd.Timedelta(minutes=15)
ATTRS = {"s15": False, "alert_score": False, "idio_24h": False, "beta": True, "r2": True, "open_n": True}   # True — менше = краще
PRIORITY = {"first": None, "s15": ("s15", False), "idio_24h": ("idio_24h", False), "low_beta": ("beta", True),
            "low_r2": ("r2", True), "random": ("rnd", False)}
LIMITS = (3, 5)
OUT = os.path.join("reports", "signal_ranking.md")


def trades() -> pd.DataFrame:
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    idx = I.market_index()
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    trained = {t.upper() for t in read_tickers_file("tickers.txt")} | set(m15.get("tickers", [])) | set(pd.read_csv("pumps.csv")["base_asset"].str.upper())
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"угоди робочої моделі: {len(coins)} монет…", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        nf = I.new_features(h, idx)
        f15 = f15.drop(columns=[k for k in nf.columns if k in f15.columns]).join(nf.reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        s15 = f15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(f15[m15["features"]])[:, 1], threshold=m15["threshold"])
        t = tier_frame(s15, s1, share).set_index("ts").sort_index()
        t = t[(t.index >= start) & (t.index < end)]
        feat = f15.set_index("ts")[["idio_24h", "beta", "r2"]]
        for r in P.simulate(hh, t["alert_score"], I.K, exit_bars, 3.0, 0.5, True):
            ts = pd.Timestamp(r["ts"])
            mth = ts.strftime("%Y-%m")
            if mth not in uni[c]:
                continue
            rows.append({"ticker": c, "month": mth, "band": uni[c][mth], **r, "exit_ts": ts + r["bars"] * BAR,
                         "s15": t.at[ts, "s15"], "alert_score": t.at[ts, "alert_score"],
                         **(feat.loc[ts].to_dict() if ts in feat.index else {})})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    d = pd.DataFrame(rows).sort_values(["ts", "ticker"]).reset_index(drop=True)
    d["ts"], d["exit_ts"] = pd.to_datetime(d["ts"], utc=True), pd.to_datetime(d["exit_ts"], utc=True)
    return d


def open_count(d: pd.DataFrame) -> np.ndarray:
    """Скільки угод (того самого набору) відкрито на момент входу — без ліміту."""
    e, x = d["ts"].values, np.sort(d["exit_ts"].values)
    return np.searchsorted(e, e, side="left") - np.searchsorted(x, e, side="right")


def limited(d: pd.DataFrame, n: int, prio) -> pd.DataFrame:
    """Ліміт n одночасних позицій; сигнали одного бару — за пріоритетом prio (колонка, менше=краще?)."""
    taken, open_exit = [], []
    for ts, g in d.groupby("ts", sort=True):
        open_exit = [x for x in open_exit if x > ts]
        free = n - len(open_exit)
        if free <= 0:
            continue
        if prio is not None:
            col, asc = prio
            g = g.sort_values(col, ascending=asc, na_position="last")
        for i, r in g.head(free).iterrows():
            taken.append(i)
            open_exit.append(r["exit_ts"])
    return d.loc[taken]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Вибір серед одночасних сигналів: квінтилі ознак входу і ліміт позицій з пріоритетом").parse_args()
    d = trades()
    d["rnd"] = np.random.default_rng(42).random(len(d))
    d["half"] = np.where(d["month"] <= "2026-06", "A", "B")
    d["open_n"] = 0
    for b in d["band"].unique():
        m = d["band"] == b
        d.loc[m, "open_n"] = open_count(d[m])
    d.to_csv(os.path.join("reports", "signal_ranking_trades.csv"), index=False)
    pf = I.pf
    big = d[d["band"] == "≥ 20 млн"].reset_index(drop=True)
    L = ["# Вибір серед одночасних сигналів", "",
         f"Угоди робочої моделі, 6 міс. архіву: {len(d)} (≥ 20 млн — {len(big)}). PF усіх {pf(d['ret']):.2f}, ≥ 20 млн {pf(big['ret']):.2f}.", "",
         "## 1. Чи сортують ознаки входу угоди (PF за квінтилями, Q1 — найменші значення)", "",
         "| ознака | набір | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 − Q1 по половинах A / B |", "|---|---|---|---|---|---|---|---|"]
    for a in ATTRS:
        for name, s in (("усі", d), ("≥ 20 млн", big)):
            s = s[s[a].notna()]
            if s[a].nunique() < 5:
                q = s[a].clip(upper=4).astype(int)
            else:
                q = pd.qcut(s[a].rank(method="first"), 5, labels=False)
            vals = [pf(s["ret"][q == k]) for k in range(5)]
            hv = []
            for hf in ("A", "B"):
                m = s["half"] == hf
                hv.append(f"{pf(s['ret'][m & (q == 4)]) - pf(s['ret'][m & (q == 0)]):+.2f}")
            L.append(f"| {a} | {name} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {' / '.join(hv)} |")
    L += ["", "open_n — кількість уже відкритих позицій (0, 1, 2, 3, 4+), не квінтилі.", "",
          "## 2. Ліміт одночасних позицій (≥ 20 млн), пріоритет серед сигналів одного бару", "",
          "| ліміт | пріоритет | угод | PF | A | B | медіана | сума |", "|---|---|---|---|---|---|---|---|"]
    r = big["ret"]
    L.append(f"| — | без ліміту | {len(big)} | {pf(r):.2f} | {pf(r[big['half'] == 'A']):.2f} | {pf(r[big['half'] == 'B']):.2f} | "
             f"{r.median():+.2%} | {r.sum():+.1f} |")
    for n in LIMITS:
        for pname, prio in PRIORITY.items():
            t = limited(big, n, prio)
            r = t["ret"]
            L.append(f"| {n} | {pname} | {len(t)} | {pf(r):.2f} | {pf(r[t['half'] == 'A']):.2f} | {pf(r[t['half'] == 'B']):.2f} | "
                     f"{r.median():+.2%} | {r.sum():+.1f} |")
    same = big.groupby("ts").size()
    L += ["", f"Барів із ≥ 2 входами ≥ 20 млн: {(same >= 2).sum()} з {len(same)} ({(same >= 2).mean():.0%}); "
          f"угод у таких барах {same[same >= 2].sum()} з {len(big)}.", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
