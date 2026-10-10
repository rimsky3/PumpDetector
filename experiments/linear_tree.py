"""
experiments/linear_tree.py — лінійні дерева LightGBM для РАНІШОГО входу (2026-10-09; продовження experiments/tree_ridge.py:
linear_tree із порогом за найкращим F1 входив, коли пройдено 12% руху проти 23% у LightGBM, при майже тому самому PF угоди,
але сигналів було в ×4.4 більше). Тепер чесне порівняння — при ОДНАКОВІЙ кількості сигналів.
Ознаки — РОБОЧОЇ моделі (MODEL_FEATURES_15M + власний рух / Amihud / волатильність), без рейнджу (власник 09.10 рейндж не взяв).
Варіанти (зафіксовано ДО запуску):
    lgbm        один LightGBM (як зараз), поріг — найкращий F1 на OOF
    lt_f1       LightGBM з лінійною моделлю в листі (linear_tree, linear_lambda 1.0), поріг — найкращий F1
    lt_match    той самий linear_tree, поріг — щоб барів із сигналом на OOF було стільки ж, скільки в lgbm
    (оцінка linear_tree — sigmoid(сира / 10): його ймовірності насичуються до 1.0, і поріг «стільки ж» інакше неможливий)
Перевірка — як tree_ridge.py: leave-ticker-out (спіймано, затримка, пройдено руху на сигналі, фальстарти, хибні) і бектест
6 міс. на чесному списку (Alert = варіант + робоча годинна).
→ reports/linear_tree.md, reports/linear_tree_trades.csv

    python experiments/linear_tree.py
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
from features import INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

COLS = MODEL_FEATURES_15M + R.MOVE
VARIANTS = ["lgbm", "lt_f1", "lt_match"]
OUT = os.path.join("reports", "linear_tree.md")


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


class LinearTreeScaled:
    """linear_tree з оцінкою sigmoid(raw / 10): ймовірності linear_tree насичуються до рівно 1.0 (лінійні листки
    екстраполюють), і поріг «стільки ж сигналів» упирався в 1.0 (перший запуск 09.10). Сира оцінка ÷ 10 — той самий
    порядок барів, але без нічиїх на стелі; tier_frame працює з будь-якою додатною оцінкою і порогом."""

    def fit(self, X, y):
        self.m = make_linear_tree().fit(X, y)
        return self

    def predict_proba(self, X):
        s = 1 / (1 + np.exp(-self.m.booster_.predict(X, raw_score=True) / 10))
        return np.column_stack([1 - s, s])


MAKERS = {"lgbm": train.make_lightgbm, "lt_f1": LinearTreeScaled, "lt_match": LinearTreeScaled}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Лінійні дерева LightGBM на ознаках робочої моделі: поріг F1 і поріг «стільки ж сигналів, як у lgbm»").parse_args()
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
    models, thr, lto, oof = {}, {}, {}, {}
    for v in VARIANTS:
        p = oof["lt"] if v == "lt_match" else train.oof_proba(MAKERS[v], X, y, grp)
        if v == "lt_f1":
            oof["lt"] = p
        if v == "lt_match":                      # стільки ж барів із сигналом, скільки в lgbm
            n_sig = int((oof["lgbm"] >= thr["lgbm"]).sum())
            thr[v] = float(np.sort(p)[::-1][max(n_sig - 1, 0)])
        else:
            thr[v] = train.best_f1_threshold(y.to_numpy(), p)
        if v == "lgbm":
            oof["lgbm"] = p
        sig = p >= thr[v]
        lto[v] = {**train.event_metrics(d, sig, pumps, train.BAR_15M), "move_done": E.move_done(d, sig, pumps, px)}
        models[v] = models["lt_f1"] if v == "lt_match" else MAKERS[v]().fit(X, y)
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
    tr.to_csv(os.path.join("reports", "linear_tree_trades.csv"), index=False)
    pf = I.pf
    L = ["# Лінійні дерева LightGBM при однаковій кількості сигналів", "", "Ознаки — робочої моделі (без рейнджу). Пороги: " +
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
