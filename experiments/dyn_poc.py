"""
experiments/dyn_poc.py — «динамічний POC» на коротких вікнах (2026-10-09, власник: «можливо тут треба зменшити таймфрейм» —
у experiments/vp_features.py POC 7-денного профілю за 24 год майже не рухався, медіана зсуву 0, ознака 0.8% ваги).

Профіль — з 15m-свічок (обсяг рівномірно по діапазону свічки, кошики 0.5%, як vp_features). Ознаки (зафіксовано ДО запуску):
    vp24_poc_dist  (ціна − POC) / ширина ковзного профілю 24 год
    vp24_shift_1h  зсув POC профілю 24 год за 1 год / ширина
    vp24_shift_4h  зсув POC профілю 24 год за 4 год / ширина
    vpd_poc_dist   (ціна − POC) / ширина профілю з початку доби UTC (як ATAS за замовчуванням; < 2 год від початку — немає)
    vpd_shift_4h   зсув POC денного профілю за 4 год / ширина (лише в межах тієї самої доби)
Варіанти: range (= base + ознаки рейнджу, найкращий на 09.10) і range_dyn (+ ці 5). Перевірка — як range_features:
leave-ticker-out на розмітці і бектест 6 міс. на чесному списку.
→ reports/dyn_poc.md, reports/dyn_poc_trades.csv

    python experiments/dyn_poc.py
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
DYN = ["vp24_poc_dist", "vp24_shift_1h", "vp24_shift_4h", "vpd_poc_dist", "vpd_shift_4h"]
RANGE_SET = MODEL_FEATURES_15M + R.MOVE + R.RANGE
VARIANTS = {"range": RANGE_SET, "range_dyn": RANGE_SET + DYN}
OUT = os.path.join("reports", "dyn_poc.md")


def _profile_cum(h: pd.DataFrame):
    hi, lo, px, v = (h[c].to_numpy(dtype=float) for c in ("high", "low", "price", "volume"))
    ok = (hi > 0) & (lo > 0) & (px > 0) & np.isfinite(v) & (v > 0)
    if not ok.any():
        return None
    g0 = int(np.floor(np.nanmin(np.log(np.where(ok, lo, np.nan))) / STEP)) - 1
    b = lambda x: np.where(ok, np.floor(np.log(np.where(ok, x, 1)) / STEP) - g0, 0).astype(int)   # noqa: E731
    ib_lo, ib_hi, jp = b(lo), b(hi), b(px)
    nb, n = int(ib_hi.max()) + 3, len(h)
    D = np.zeros((n, nb + 1), dtype=np.float32)
    w = np.where(ok, v / (ib_hi - ib_lo + 1), 0)
    np.add.at(D, (np.arange(n), ib_lo), w)
    np.add.at(D, (np.arange(n), ib_hi + 1), -w)
    C = np.vstack([np.zeros((1, nb)), np.cumsum(np.cumsum(D, axis=1)[:, :nb], axis=0, dtype=np.float64)])
    return C, jp, ok, nb


def _poc_width(Hw: np.ndarray):
    poc = Hw.argmax(axis=1)
    occ = Hw > 0
    first = occ.argmax(axis=1)
    last = Hw.shape[1] - 1 - occ[:, ::-1].argmax(axis=1)
    return poc.astype(float), np.maximum(last - first, 1).astype(float), Hw.sum(axis=1) > 0


def dyn_features(h: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=h.index, columns=DYN)
    pc = _profile_cum(h)
    if pc is None:
        return out
    C, jp, ok, nb = pc
    n = len(h)
    # ковзний профіль 24 год (96 барів, включно з поточним)
    if n > 96:
        t = np.arange(96, n)
        poc, wid, val = _poc_width(C[t + 1] - C[t + 1 - 96])
        val &= ok[t]
        dist = np.where(val, (jp[t] - poc) / wid, np.nan)
        s1, s4 = np.full(len(t), np.nan), np.full(len(t), np.nan)
        s1[4:] = (poc[4:] - poc[:-4]) / wid[4:]
        s4[16:] = (poc[16:] - poc[:-16]) / wid[16:]
        out.iloc[t, 0], out.iloc[t, 1], out.iloc[t, 2] = dist, np.where(val, s1, np.nan), np.where(val, s4, np.nan)
    # профіль з початку доби UTC
    day = h.index.floor("D")
    start = pd.Series(np.arange(n), index=h.index).groupby(day).transform("min").to_numpy()
    t = np.arange(n)
    poc, wid, val = _poc_width(C[t + 1] - C[start])
    val &= ok & (t - start >= 8)
    out.iloc[:, 3] = np.where(val, (jp - poc) / wid, np.nan)
    s4 = np.full(n, np.nan)
    same = (t >= 16) & (start == np.roll(start, 16))
    s4[same] = (poc[same] - poc[np.where(same)[0] - 16]) / wid[same]
    out.iloc[:, 4] = np.where(val, s4, np.nan)
    return out.astype(float)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Динамічний POC на коротких вікнах (24 год, з початку доби) поверх ознак рейнджу").parse_args()
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
        nf = I.new_features(a, idx).join(R.range_features(a, a["oi_coin_log_idx"])).join(dyn_features(a))
        parts.append(nf.reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts)).sort_values(["base_asset", "ts"]).reset_index(drop=True)
    y = d["label"].astype(int)
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів; динамічний POC 24 год є в {d['vp24_shift_4h'].notna().mean():.0%}, денний — в "
          f"{d['vpd_shift_4h'].notna().mean():.0%} барів", flush=True)
    models, lto, imp = {}, {}, None
    for name, cols in VARIANTS.items():
        p = train.oof_proba(train.make_lightgbm, d[cols], y, d["base_asset"])
        thr = train.best_f1_threshold(y.to_numpy(), p)
        lto[name] = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        mo = train.make_lightgbm().fit(d[cols], y)
        models[name] = (mo, thr, cols)
        if name == "range_dyn":
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
        nf = I.new_features(h, idx).join(R.range_features(h, R.archive_oi_log(c, h.index))).join(dyn_features(h))
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
    tr.to_csv(os.path.join("reports", "dyn_poc_trades.csv"), index=False)
    pf, names = I.pf, list(VARIANTS)
    L = ["# Динамічний POC на коротких вікнах поверх ознак рейнджу", "",
         "range = ознаки робочої моделі + рейндж; range_dyn = + " + ", ".join(DYN) + ". Обидві свіжонавчені однаково.", "",
         "## Leave-ticker-out (Кандидат 15m)", "", "| варіант | спіймано | затримка, год | фальстартів | хибних / 30 дн |", "|---|---|---|---|---|"]
    for v, m in lto.items():
        L.append(f"| {v} | {m['pumps_caught']} | {m['median_delay_h']:.2f} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест 6 міс., {tr['ticker'].nunique()} монет чесного списку", "",
          "| варіант | угод | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|"]
    for v in names:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                 f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[names]
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[names]
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | range | range_dyn |", "|---|---|---|"]
    for mth in piv.index:
        L.append(f"| {mth} | {piv.at[mth, 'range']:.2f} ({piv20.at[mth, 'range']:.2f}) | {piv.at[mth, 'range_dyn']:.2f} ({piv20.at[mth, 'range_dyn']:.2f}) |")
    L += ["", f"Місяців, де range_dyn кращий: усі {int((piv['range_dyn'] > piv['range'] + 1e-9).sum())}, ≥ 20 млн "
          f"{int((piv20['range_dyn'] > piv20['range'] + 1e-9).sum())} з {len(piv)}", ""]
    if imp is not None:
        L += [f"Важливість динамічного POC разом {imp[DYN].sum():.1%}: " + ", ".join(f"{k} {imp[k]:.1%}" for k in DYN), "",
              "Топ-10: " + ", ".join(f"{k} {v:.1%}" for k, v in imp.sort_values(ascending=False).head(10).items()), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
