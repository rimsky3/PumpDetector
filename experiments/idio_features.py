"""
experiments/idio_features.py — нові ознаки 15m-моделі на 6 місяцях і ЧЕСНОМУ списку монет (2026-10-08).

Точка відліку — long_binance.py --universe monthly: робоча модель на монетах «станом на кожен місяць» (з делістованими),
PF 0.89. Три групи ознак (пояснення — notes/movers.md, ті самі метрики, що в кнопці «Movers», але на 15m-барах):
    IDIO  «власний» рух: beta / r2 до ALT-EW (рівноважений індекс усіх монет кешу; ковзні 7 днів 15m-доходностей, вікно
          закінчується за добу до бару), idio_1h / 4h / 24h = доходність − β × індекс, idio_z_4h = idio_4h у σ залишків
    AMI   Amihud: illiq_x = log(середнє |r| / $-обіг за 24 год ÷ медіана цього за 30 днів до того)
    VOL   park_24h / gk_24h (волатильність Паркінсона / Гарман–Класа за 24 год), park_x (до медіани 7 днів),
          ret_4h_vol / ret_24h_vol = доходність ÷ (волатильність Паркінсона × √барів)
Варіанти: base (MODEL_FEATURES_15M), +idio, +ami, +vol, +all — моделі свіжонавчені однаково (тікери навчання, лише Binance,
як long_binance.py). Угоди — Alert (15m + робоча годинна на власних 15m Binance), частковий вихід, стоп 2.48 ATR, лише в
місяцях, коли монета була в списку. Критерій — «краще за base у більшості з 6 місяців», не загальний PF.
Без погляду вперед: усі вікна закінчуються на барі (ковзні назад), бета — ще й зі зсувом на добу. Навчальні бари після
30.09 (кінець кешу індексу) мають NaN у нових ознаках — LightGBM уміє.

    python experiments/idio_features.py
"""

import argparse
import glob
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
import train
from features import INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

DIR = os.path.join("data", "long_bin")
START, END = "2026-04-01", "2026-09-30"
K = 2.48
B = 96                                      # 15m-барів на добу
IDIO = ["beta", "r2", "idio_1h", "idio_4h", "idio_24h", "idio_z_4h"]
AMI = ["illiq_x"]
VOL = ["park_24h", "gk_24h", "park_x", "ret_4h_vol", "ret_24h_vol"]
VARIANTS = {"base": [], "idio": IDIO, "ami": AMI, "vol": VOL, "all": IDIO + AMI + VOL}
OUT = os.path.join("reports", "idio_features.md")


def market_index() -> pd.Series:
    """ALT-EW на 15m: середня log-доходність усіх монет кешу (≥ 50 з даними на барі)."""
    rets = {}
    for p in glob.glob(os.path.join(DIR, "*.pkl")):
        h = pickle.load(open(p, "rb"))["h"]
        rets[os.path.basename(p)[:-4]] = np.log(h["price"].where(h["price"] > 0)).diff()
    w = pd.DataFrame(rets)
    return w.mean(axis=1).where(w.notna().sum(axis=1) >= 50)


def new_features(h: pd.DataFrame, idx: pd.Series) -> pd.DataFrame:
    """15m-ряд монети (open, high, low, price, volume у $) + індекс → нові ознаки на кожному барі (лише минуле)."""
    f = pd.DataFrame(index=h.index)
    lr = np.log(h["price"].where(h["price"] > 0)).diff()
    ix = idx.reindex(h.index)
    W = 7 * B
    m = pd.concat([lr, ix], axis=1, keys=["y", "x"]).dropna()
    ry, rx = m["y"].reindex(h.index), m["x"].reindex(h.index)
    mean = lambda s: s.rolling(W, min_periods=W // 2).mean()      # noqa: E731
    cov = (mean(ry * rx) - mean(ry) * mean(rx)).shift(B)
    vx = (mean(rx * rx) - mean(rx) ** 2).shift(B)
    vy = (mean(ry * ry) - mean(ry) ** 2).shift(B)
    beta = cov / vx.where(vx > 0)
    f["beta"] = beta
    f["r2"] = (cov ** 2 / (vx * vy)).where((vx > 0) & (vy > 0))
    resid = (vy - beta ** 2 * vx).clip(lower=1e-12)
    for k, name in ((4, "idio_1h"), (16, "idio_4h"), (B, "idio_24h")):
        f[name] = lr.rolling(k).sum() - beta * ix.rolling(k).sum()
    f["idio_z_4h"] = f["idio_4h"] / np.sqrt(resid * 16)
    il = lr.abs() / h["volume"].where(h["volume"] > 0)
    roll = il.rolling(B, min_periods=72).mean()
    norm = roll.shift(B).rolling(30 * B, min_periods=7 * B).median()
    f["illiq_x"] = np.log(roll / norm.where(norm > 0))
    hl = np.log(h["high"] / h["low"]) ** 2
    co = np.log(h["price"] / h["open"]) ** 2
    park = np.sqrt((hl / (4 * np.log(2))).rolling(B, min_periods=72).mean())
    gk = np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).rolling(B, min_periods=72).mean().clip(lower=0))
    f["park_24h"], f["gk_24h"] = park, gk
    f["park_x"] = park / park.shift(B).rolling(7 * B, min_periods=3 * B).median()
    f["ret_4h_vol"] = lr.rolling(16).sum() / (park * 4)
    f["ret_24h_vol"] = lr.rolling(B).sum() / (park * np.sqrt(B))
    return f.replace([np.inf, -np.inf], np.nan)


def pf(v) -> float:
    v = pd.Series(v)
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ознаки власного руху / Amihud / волатильності в 15m-моделі: 6 міс., чесний список монет").parse_args()
    load_dotenv(".env")
    uni = json.load(open(os.path.join(DIR, "_universe.json"), encoding="utf-8"))
    print("індекс ALT-EW з кешу…", flush=True)
    idx = market_index()
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
        nf = new_features(aggregate(raw, b, "15min"), idx).reindex(g["ts"]).set_axis(g.index)
        parts.append(nf)
    d = d.join(pd.concat(parts))
    y = d["label"].astype(int)
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів; нові ознаки є в {d['idio_4h'].notna().mean():.0%} барів", flush=True)
    models, imp = {}, None
    for name, extra in VARIANTS.items():
        cols = MODEL_FEATURES_15M + extra
        p = train.oof_proba(train.make_lightgbm, d[cols], y, d["base_asset"])
        mo = train.make_lightgbm().fit(d[cols], y)
        models[name] = (mo, train.best_f1_threshold(y.to_numpy(), p), cols)
        if name == "all":
            g = pd.Series(mo.booster_.feature_importance("gain"), index=cols)
            imp = g / g.sum()
        print(f"  модель {name}: {len(cols)} ознак", flush=True)

    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    exit_bars, share = m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    start, end = pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1)
    rows = []
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(DIR, f"{c}.pkl")))
    print(f"бектест: {len(coins)} монет чесного списку…", flush=True)
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        months = uni[c]
        for name, (mo, thr, cols) in models.items():
            s15 = f15[["base_asset", "ts"]].assign(probability=mo.predict_proba(f15[cols])[:, 1], threshold=thr)
            t = tier_frame(s15, s1, share).set_index("ts").sort_index()
            sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
            for r in P.simulate(hh, sc, K, exit_bars, 3.0, 0.5, True):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": name, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join("reports", "idio_features_trades.csv"), index=False)
    lines = ["# Нові ознаки 15m-моделі: власний рух / Amihud / волатильність", "",
             f"{tr['ticker'].nunique()} монет чесного списку (станом на кожен місяць), {START} – {END}; лише Binance; Alert = 15m + "
             f"робоча годинна ≥ {share}; частковий вихід, стоп {K} ATR, комісія 0.1%. Точка відліку — base.", "",
             "## Загалом", "", "| варіант | угод | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        lines.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                     f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[list(VARIANTS)]
    lines += ["", "## По місяцях (PF)", "", "| місяць | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for m, r in piv.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{r[v]:.2f}" for v in VARIANTS) + " |")
    lines += ["", "Місяців, де варіант кращий за base: " + ", ".join(f"{v} {int((piv[v] > piv['base'] + 1e-9).sum())} з {len(piv)}" for v in VARIANTS if v != "base"), ""]
    if imp is not None:
        lines += ["## Важливість у варіанті all (gain)", "",
                  "Групи: " + ", ".join(f"{n} {imp[g].sum():.1%}" for n, g in (("IDIO", IDIO), ("AMI", AMI), ("VOL", VOL))), "",
                  "Топ-10: " + ", ".join(f"{k} {v:.1%}" for k, v in imp.sort_values(ascending=False).head(10).items()), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {OUT}")


if __name__ == "__main__":
    main()
