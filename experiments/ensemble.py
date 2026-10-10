"""
experiments/ensemble.py — ансамблі 15m-моделей (2026-10-09, власник: «те, що раніше ловить логрегресія, фільтрувати через
LightGBM — щоб заходити раніше»; і загалом практики ансамблів). Набір ознак — найкращий на 09.10: ознаки робочої моделі +
рейндж (experiments/range_features.py). Варіанти (зафіксовано ДО запуску):
    lgbm        один LightGBM (як зараз) — точка відліку
    lgbm_x5     середнє 5 LightGBM з різним зерном (random_state 42…46; кожен вчиться на 80% рядків і ознак)
    logreg      логістична регресія (L2, як train.make_logreg)
    blend       середнє «впевненостей» lgbm і logreg (кожна ÷ свій поріг); поріг блендa — найкращий F1 на OOF
    lr_lgbm50   logreg ≥ свого порогу І lgbm ≥ 0.5 свого порогу (рання модель + м'яке підтвердження)
    lr_lgbm75   те саме з 0.75
Пороги — найкращий F1 на OOF кожної моделі (однаково для всіх). Перевірка 1 — leave-ticker-out на розмітці (Кандидат 15m):
спіймано, затримка, пройдено руху на першому сигналі ((ціна − старт) / (пік − старт), медіана), фальстарти, хибні на 30 дн.
Перевірка 2 — бектест 6 міс. на чесному списку: Alert = варіант 15m + робоча годинна ≥ 0.75, частковий вихід, стоп 2.48 ATR.
→ reports/ensemble.md, reports/ensemble_trades.csv

    python experiments/ensemble.py
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

import exit_partial as P
import idio_features as I
import range_features as R
import train
from features import INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, in_pump, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

COLS = MODEL_FEATURES_15M + R.MOVE + R.RANGE
SEEDS = (42, 43, 44, 45, 46)
VARIANTS = ["lgbm", "lgbm_x5", "logreg", "blend", "lr_lgbm50", "lr_lgbm75"]
OUT = os.path.join("reports", "ensemble.md")


def lgbm(seed: int):
    m = train.make_lightgbm()
    m.set_params(random_state=seed)
    return m


def scores(p_l: np.ndarray, p_x5: np.ndarray, p_r: np.ndarray, thr: dict) -> dict[str, np.ndarray]:
    """Оцінка кожного варіанта, нормована так, що сигнал = оцінка ≥ 1."""
    sl, sr = p_l / thr["lgbm"], p_r / thr["logreg"]
    return {"lgbm": sl, "lgbm_x5": p_x5 / thr["lgbm_x5"], "logreg": sr,
            "blend": (sl + sr) / 2 / thr["blend"],
            "lr_lgbm50": np.minimum(sr, sl / 0.5), "lr_lgbm75": np.minimum(sr, sl / 0.75)}


def move_done(d: pd.DataFrame, sig: np.ndarray, pumps: pd.DataFrame, px: dict) -> float:
    """Медіана частки руху, пройденої на першому сигналі всередині імпульсу (лише спіймані)."""
    out = []
    s = d.loc[sig, ["base_asset", "ts"]]
    for r in pumps[pumps["pattern_type"] == "pump"].itertuples():
        h = px.get(r.base_asset)
        if h is None:
            continue
        ts = s[(s["base_asset"] == r.base_asset)]["ts"]
        bar_open = ts[in_pump(ts, r.start_ts, r.end_ts, train.BAR_15M)] if len(ts) else ts
        if not len(bar_open):
            continue
        w = h[(h.index >= r.start_ts - train.BAR_15M) & (h.index < r.end_ts)]
        if w.empty:
            continue
        p0 = h["price"][h.index < r.start_ts].iloc[-1] if (h.index < r.start_ts).any() else w["price"].iloc[0]
        peak = w["high"].max()
        p1 = h.at[bar_open.min(), "price"] if bar_open.min() in h.index else np.nan
        if peak > p0 and np.isfinite(p1):
            out.append((p1 - p0) / (peak - p0))
    return float(np.median(out)) if out else np.nan


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ансамблі 15m-моделей: LightGBM ×5 зерен, логрегресія, бленд, рання модель + підтвердження").parse_args()
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
    yy = y.to_numpy()
    p_l = train.oof_proba(lambda: lgbm(42), X, y, grp)
    print("  lgbm OOF готово", flush=True)
    p_seeds = [p_l] + [train.oof_proba(lambda s=s: lgbm(s), X, y, grp) for s in SEEDS[1:]]
    p_x5 = np.mean(p_seeds, axis=0)
    print("  lgbm ×5 OOF готово", flush=True)
    p_r = train.oof_proba(train.make_logreg, X, y, grp)
    thr = {"lgbm": train.best_f1_threshold(yy, p_l), "lgbm_x5": train.best_f1_threshold(yy, p_x5),
           "logreg": train.best_f1_threshold(yy, p_r)}
    thr["blend"] = train.best_f1_threshold(yy, (p_l / thr["lgbm"] + p_r / thr["logreg"]) / 2)
    sc = scores(p_l, p_x5, p_r, thr)
    lto = {}
    for v in VARIANTS:
        sig = sc[v] >= 1
        lto[v] = {**train.event_metrics(d, sig, pumps, train.BAR_15M), "move_done": move_done(d, sig, pumps, px)}
        print(f"  {v}: {lto[v]}", flush=True)
    models = {"lgbm": [lgbm(42).fit(X, y)], "logreg": train.make_logreg().fit(X, y)}
    models["x5"] = [models["lgbm"][0]] + [lgbm(s).fit(X, y) for s in SEEDS[1:]]

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
        Xc = f15[COLS]
        ql = models["lgbm"][0].predict_proba(Xc)[:, 1]
        qx = np.mean([m.predict_proba(Xc)[:, 1] for m in models["x5"]], axis=0)
        qr = models["logreg"].predict_proba(Xc)[:, 1]
        sc_c = scores(ql, qx, qr, thr)
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        months = uni[c]
        for v in VARIANTS:
            s15 = f15[["base_asset", "ts"]].assign(probability=sc_c[v], threshold=1.0)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            a_sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
            for r in P.simulate(hh, a_sc, I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": v, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join("reports", "ensemble_trades.csv"), index=False)
    pf = I.pf
    L = ["# Ансамблі 15m-моделей", "", "Ознаки — робочі + рейндж. Пороги — найкращий F1 на OOF: " +
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
