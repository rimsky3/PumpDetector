"""
experiments/vp_features.py — ознаки профілю обсягу В 15m-МОДЕЛІ (2026-10-09, власник: «можливо як ознаку додати профіль об'єму»).
Раніше (03.10) рівні профілю дня як рівні відскоку ≈ випадкові; тут інше питання — чи допомагає профіль відрізнити імпульс,
що піде далі, від фальстарту: «тиск зверху» (обсяг, проторгований вище ціни) і «порожнеча попереду».

Профіль — з 15m-свічок: обсяг ($) свічки рівномірно по її діапазону low–high, кошики 0.5% (лог-ціна), вікно — останні 7 днів
(672 бари, включно з поточним), для vp30 — 30 днів. Ознаки (зафіксовано ДО запуску):
    vp_above_10   частка обсягу 7 дн у кошиках від ціни (не включно) до +10% над нею
    vp_above_all  частка обсягу 7 дн у всіх кошиках вище ціни
    vp30_above_10 як vp_above_10, але за 30 днів
    vp_thin_3     обсяг у +0…3% над ціною на кошик ÷ середній обсяг на непорожній кошик профілю 7 дн (мало = порожнеча)
    vp_va_pos     (ціна − VAL) / (VAH − VAL), VAL / VAH — 15-й / 85-й перцентиль обсягу профілю 7 дн (> 1 — над зоною вартості)
    vp_poc_dist   (ціна − POC) / ширина профілю 7 дн (лог)
    vp_poc_shift  зсув POC профілю 7 дн за 24 год / ширина профілю — «динамічний POC» (ATAS Dynamic Levels, власник 09.10):
                  пробій, за яким POC мігрує вгору, «прийнятий»; POC лишився внизу — ціну вище ще не прийняли
Варіанти: base (ознаки робочої моделі, як experiments/range_features.py), vp = base + VP, range_vp = base + рейндж + VP.
Перевірка та сама, що в range_features: leave-ticker-out на розмітці (Кандидат 15m) і бектест 6 міс. на чесному списку.
→ reports/vp_features.md, reports/vp_features_trades.csv

    python experiments/vp_features.py
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
from features import INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

STEP = np.log(1.005)
VP = ["vp_above_10", "vp_above_all", "vp30_above_10", "vp_thin_3", "vp_va_pos", "vp_poc_dist", "vp_poc_shift"]
BASE = MODEL_FEATURES_15M + R.MOVE
VARIANTS = {"base": BASE, "vp": BASE + VP, "range_vp": BASE + R.RANGE + VP}
OUT = os.path.join("reports", "vp_features.md")


def vp_features(h: pd.DataFrame) -> pd.DataFrame:
    """15m-ряд (high, low, price, volume $) → ознаки профілю обсягу на кожному барі (вікно закінчується на ньому)."""
    hi, lo, px, v = (h[c].to_numpy(dtype=float) for c in ("high", "low", "price", "volume"))
    ok = (hi > 0) & (lo > 0) & (px > 0) & np.isfinite(v) & (v > 0)
    lp = np.log(np.where(ok, np.r_[lo], np.nan))
    if not ok.any():
        return pd.DataFrame(index=h.index, columns=VP, dtype=float)
    g0 = int(np.floor(np.nanmin(lp) / STEP)) - 1
    ib_lo = np.where(ok, np.floor(np.log(np.where(ok, lo, 1)) / STEP) - g0, 0).astype(int)
    ib_hi = np.where(ok, np.floor(np.log(np.where(ok, hi, 1)) / STEP) - g0, 0).astype(int)
    nb = int(ib_hi.max()) + 3
    n = len(h)
    D = np.zeros((n, nb + 1), dtype=np.float32)             # різниці: рівномірний розподіл обсягу бару по кошиках
    w = np.where(ok, v / (ib_hi - ib_lo + 1), 0)
    rows = np.arange(n)
    np.add.at(D, (rows, ib_lo), w)
    np.add.at(D, (rows, ib_hi + 1), -w)
    bar_hist = np.cumsum(D, axis=1)[:, :nb]
    C = np.vstack([np.zeros((1, nb), dtype=np.float64), np.cumsum(bar_hist, axis=0, dtype=np.float64)])
    jp = np.where(ok, np.floor(np.log(np.where(ok, px, 1)) / STEP) - g0, 0).astype(int)
    k10, k3 = int(round(np.log(1.10) / STEP)), int(round(np.log(1.03) / STEP))
    out = pd.DataFrame(index=h.index, columns=VP, dtype=float)
    for W, cols in ((672, ("vp_above_10", "vp_above_all", "vp_thin_3", "vp_va_pos", "vp_poc_dist", "vp_poc_shift")), (2880, ("vp30_above_10",))):
        if n <= W:
            continue
        t = np.arange(W, n)                                     # бари з повним вікном [t − W + 1, t]
        Hw = C[t + 1] - C[t + 1 - W]                            # профіль вікна, (рядки × кошики)
        tot = Hw.sum(axis=1)
        cb = np.cumsum(Hw, axis=1)
        j = np.clip(jp[t], 0, nb - 1)
        r = np.arange(len(t))
        upto = cb[r, j]
        j10 = np.clip(j + k10, 0, nb - 1)
        valid = (tot > 0) & ok[t]
        if "vp_above_10" in cols:
            out.iloc[t, out.columns.get_loc("vp_above_10")] = np.where(valid, (cb[r, j10] - upto) / tot, np.nan)
            out.iloc[t, out.columns.get_loc("vp_above_all")] = np.where(valid, (tot - upto) / tot, np.nan)
            j3 = np.clip(j + k3, 0, nb - 1)
            occ = (Hw > 0).sum(axis=1)
            dens = tot / np.maximum(occ, 1)
            out.iloc[t, out.columns.get_loc("vp_thin_3")] = np.where(valid, (cb[r, j3] - upto) / max(k3, 1) / dens, np.nan)
            q15 = (cb >= 0.15 * tot[:, None]).argmax(axis=1)
            q85 = (cb >= 0.85 * tot[:, None]).argmax(axis=1)
            out.iloc[t, out.columns.get_loc("vp_va_pos")] = np.where(valid & (q85 > q15), (jp[t] - q15) / np.maximum(q85 - q15, 1), np.nan)
            poc = Hw.argmax(axis=1)
            first = (Hw > 0).argmax(axis=1)
            last = nb - 1 - (Hw[:, ::-1] > 0).argmax(axis=1)
            out.iloc[t, out.columns.get_loc("vp_poc_dist")] = np.where(valid & (last > first), (jp[t] - poc) / np.maximum(last - first, 1), np.nan)
            shift = np.full(len(t), np.nan)
            shift[96:] = (poc[96:] - poc[:-96]) / np.maximum(last[96:] - first[96:], 1)     # POC зараз − POC 24 год тому
            out.iloc[t, out.columns.get_loc("vp_poc_shift")] = np.where(valid, shift, np.nan)
        else:
            out.iloc[t, out.columns.get_loc("vp30_above_10")] = np.where(valid, (cb[r, j10] - upto) / tot, np.nan)
    return out.astype(float)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ознаки профілю обсягу в 15m-моделі: фальстарти (LTO) і бектест 6 міс. на чесному списку").parse_args()
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
    parts = []
    for b, g in d.groupby("base_asset"):
        a = aggregate(raw, b, "15min")
        nf = I.new_features(a, idx).join(R.range_features(a, a["oi_coin_log_idx"])).join(vp_features(a))
        parts.append(nf.reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts)).sort_values(["base_asset", "ts"]).reset_index(drop=True)
    y = d["label"].astype(int)
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів; профіль 7 дн є в {d['vp_above_10'].notna().mean():.0%}, "
          f"30 дн — в {d['vp30_above_10'].notna().mean():.0%} барів", flush=True)
    models, lto, imp = {}, {}, None
    for name, cols in VARIANTS.items():
        p = train.oof_proba(train.make_lightgbm, d[cols], y, d["base_asset"])
        thr = train.best_f1_threshold(y.to_numpy(), p)
        lto[name] = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        mo = train.make_lightgbm().fit(d[cols], y)
        models[name] = (mo, thr, cols)
        if name == "range_vp":
            g = pd.Series(mo.booster_.feature_importance("gain"), index=cols)
            imp = g / g.sum()
        print(f"  {name}: {len(cols)} ознак, LTO {lto[name]}", flush=True)

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
        nf = I.new_features(h, idx).join(R.range_features(h, R.archive_oi_log(c, h.index))).join(vp_features(h))
        f15 = f15.join(nf.reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        months = uni[c]
        for name, (mo, thr, cols) in models.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=mo.predict_proba(f15[cols])[:, 1], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
            for r in P.simulate(hh, sc, I.K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": name, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join("reports", "vp_features_trades.csv"), index=False)
    pf = I.pf
    names = list(VARIANTS)
    L = ["# Ознаки профілю обсягу в 15m-моделі", "",
         "Усі варіанти свіжонавчені однаково (тікери навчання, лише Binance). base = ознаки робочої моделі, vp = base + " + ", ".join(VP)
         + ", range_vp = base + ознаки рейнджу + профіль.", "",
         "## Leave-ticker-out на розмітці (Кандидат = лише 15m, поріг — найкращий F1 на OOF)", "",
         "| варіант | спіймано імпульсів | затримка, год | фальстартів спрацювало | хибних / 30 дн |", "|---|---|---|---|---|"]
    for v, m in lto.items():
        L.append(f"| {v} | {m['pumps_caught']} | {m['median_delay_h']:.2f} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест: {tr['ticker'].nunique()} монет чесного списку, {I.START} – {I.END}, Alert, частковий вихід, стоп {I.K} ATR", "",
          "| варіант | угод | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|"]
    for v in names:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                 f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[names]
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[names]
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for mth in piv.index:
        L.append(f"| {mth} | " + " | ".join(f"{piv.at[mth, v]:.2f} ({piv20.at[mth, v]:.2f})" for v in names) + " |")
    L += ["", "Місяців кращих за base: " + ", ".join(
        f"{v} усі {int((piv[v] > piv['base'] + 1e-9).sum())} / ≥ 20 млн {int((piv20[v] > piv20['base'] + 1e-9).sum())} з {len(piv)}"
        for v in names if v != "base"), ""]
    if imp is not None:
        L += ["## Важливість у варіанті range_vp (gain)", "",
              f"Профіль разом {imp[VP].sum():.1%} — " + ", ".join(f"{k} {imp[k]:.1%}" for k in VP) + f"; рейндж разом {imp[R.RANGE].sum():.1%}", "",
              "Топ-10: " + ", ".join(f"{k} {v:.1%}" for k, v in imp.sort_values(ascending=False).head(10).items()), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
