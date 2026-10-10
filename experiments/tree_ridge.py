"""
experiments/tree_ridge.py — «дерева + ridge» (2026-10-09, власник: твіт quantymacro «RenTech tree model» / «Random Ridge Forest»;
сам трюк автор не розкрив, перевіряємо найімовірнішу відому форму). Набір ознак — робочі + рейндж (як experiments/ensemble.py).
Варіанти (зафіксовано ДО запуску):
    lgbm         один LightGBM (як зараз) — точка відліку
    leaf_ridge   LightGBM → листя кожного дерева (one-hot, 300 дерев × ≤ 7 листків) → логістична регресія з L2
                 (C = 0.1, class_weight balanced, як train.make_logreg): ваги листя підбираються разом і з регуляризацією
                 (ідея RuleFit / leaf embeddings)
    linear_tree  LightGBM з лінійною моделлю в кожному листі (linear_tree, linear_lambda = 1.0), інші параметри як у робочої
Пороги — найкращий F1 на OOF. Перевірка — як ensemble.py: leave-ticker-out (спіймано, затримка, пройдено руху, фальстарти,
хибні) і бектест 6 міс. на чесному списку (Alert = варіант + робоча годинна).
→ reports/tree_ridge.md, reports/tree_ridge_trades.csv

    python experiments/tree_ridge.py
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
import psycopg
from dotenv import load_dotenv
from scipy import sparse
from sklearn.linear_model import LogisticRegression

import ensemble as E
import exit_partial as P
import idio_features as I
import range_features as R
import train
from features import INTERVAL_15M, aggregate, build_from_raw, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

COLS = E.COLS
VARIANTS = ["lgbm", "leaf_ridge", "linear_tree"]
OUT = os.path.join("reports", "tree_ridge.md")


class LeafRidge:
    """LightGBM → one-hot листя → логістична регресія з L2 (sklearn-подібний: fit / predict_proba)."""

    def fit(self, X, y):
        self.gb = train.make_lightgbm().fit(X, y)
        L = self.gb.booster_.predict(X, pred_leaf=True)
        self.n_leaf = int(L.max()) + 1
        self.lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit(self._onehot(L), y)
        return self

    def _onehot(self, L: np.ndarray):
        n, t = L.shape
        cols = (np.arange(t) * self.n_leaf)[None, :] + np.clip(L, 0, self.n_leaf - 1)
        return sparse.csr_matrix((np.ones(n * t), (np.repeat(np.arange(n), t), cols.ravel())), shape=(n, t * self.n_leaf))

    def predict_proba(self, X):
        return self.lr.predict_proba(self._onehot(self.gb.booster_.predict(X, pred_leaf=True)))


def make_linear_tree():
    m = train.make_lightgbm()
    m.set_params(linear_tree=True, linear_lambda=1.0)
    return m


MAKERS = {"lgbm": train.make_lightgbm, "leaf_ridge": LeafRidge, "linear_tree": make_linear_tree}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Дерева + ridge: листя LightGBM → логістична L2 і лінійні дерева LightGBM").parse_args()
    load_dotenv(".env")
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    print("індекс ALT-EW з кешу…", flush=True)
    idx = I.market_index()
    excluded = excluded_tickers(".")
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")} - excluded
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        pumps = pumps[~pumps["base_asset"].isin(excluded)]
        bases = sorted(reviewed | set(pumps["base_asset"]))
        raw = load_raw(conn, bases, INTERVAL_15M, exchanges=["Binance"])
        d = build_from_raw(raw, pumps, bars_per_hour=4, extra=short_features)
        first1 = query_df(conn, "SELECT base_asset, min(ts) AS t FROM features GROUP BY 1").set_index("base_asset")["t"]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
    d = d[d["base_asset"].isin(reviewed) | d["pattern_type"].notna()]
    d = d[(d["ts"] <= cutoff) | d["pattern_type"].notna()]
    d = d[d["ts"] >= d["base_asset"].map(first1)].reset_index(drop=True)
    parts, px = [], {}
    for b, g in d.groupby("base_asset"):
        a = aggregate(raw, b, "15min")
        px[b] = a[["price", "high"]]
        nf = I.new_features(a, idx).join(R.range_features(a, a["oi_coin_log_idx"]))
        parts.append(nf.reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts)).sort_values(["base_asset", "ts"]).reset_index(drop=True)
    X, y, grp = d[COLS], d["label"].astype(int), d["base_asset"]
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів", flush=True)
    models, thr, lto = {}, {}, {}
    for v in VARIANTS:
        p = train.oof_proba(MAKERS[v], X, y, grp)
        thr[v] = train.best_f1_threshold(y.to_numpy(), p)
        sig = p >= thr[v]
        lto[v] = {**train.event_metrics(d, sig, pumps, train.BAR_15M), "move_done": E.move_done(d, sig, pumps, px)}
        models[v] = MAKERS[v]().fit(X, y)
        print(f"  {v}: {lto[v]}", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    rows = []
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет чесного списку…", flush=True)
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        nf = I.new_features(h, idx).join(R.range_features(h, R.archive_oi_log(c, h.index)))
        f15 = f15.join(nf.reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        months = uni[c]
        for v in VARIANTS:
            s15 = f15[["base_asset", "ts"]].assign(probability=models[v].predict_proba(f15[COLS])[:, 1], threshold=thr[v])
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            a_sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
            for r in P.simulate(hh, a_sc, I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": v, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join("reports", "tree_ridge_trades.csv"), index=False)
    pf = I.pf
    L = ["# Дерева + ridge (leaf_ridge) і лінійні дерева LightGBM", "", "Ознаки — робочі + рейндж. Пороги — найкращий F1 на OOF: " +
         ", ".join(f"{k} {v:.3f}" for k, v in thr.items()) + ".", "",
         "## Leave-ticker-out (Кандидат 15m)", "",
         "| варіант | спіймано | затримка, год | пройдено руху на сигналі | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        m = lto[v]
        L.append(f"| {v} | {m['pumps_caught']} | {m['median_delay_h']:.2f} | {m['move_done']:.0%} | {m['false_starts_triggered']} | "
                 f"{m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест 6 міс., {tr['ticker'].nunique()} монет чесного списку (Alert = варіант + робоча годинна)", "",
          "| варіант | угод | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                 f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[VARIANTS]
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[VARIANTS]
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in VARIANTS) + " |")
    L += ["", "Місяців кращих за lgbm: " + ", ".join(
        f"{v} {int((piv[v] > piv['lgbm'] + 1e-9).sum())} / ≥ 20 млн {int((piv20[v] > piv20['lgbm'] + 1e-9).sum())}"
        for v in VARIANTS if v != "lgbm") + f" (з {len(piv)})", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
