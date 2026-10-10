"""
experiments/range_v2.py — рейндж «як бачить власник» (2026-10-09). У range_features.py вікно було фіксованим (96 год, відступ
4 год, хай / лоу по тінях), і користь дала лише brk_rng (ціна відносно хаю), решта ~1%. Власник: головне — ТРИВАЛІСТЬ проторговки
(MAGIC ~2 дні, SAND ~3, STRK ~4, бувають тижні); рамку малює по основній масі свічок, без проколів.

Визначення (зафіксовано ДО запуску):
    коридор — годинні ЗАКРИТТЯ (тіні не враховуються), межі = макс / мін закриття вікна; рейндж, якщо (верх / низ − 1) ≤ смуги
    (спершу були 5-й / 95-й перцентилі — на SAND 02.10 вони «ковтали» перші години пробою, і рейндж не закінчувався),
    смуга = max(4%, 1.5 × медіана денного діапазону (хай / лоу − 1) за 30 завершених днів);
    на кожній годині — найдовша тривалість L із {12, 24, 48, 72, 96, 144, 192, 288, 384, 504} год, для якої коридор
    вікна, що закінчується на цій годині, — рейндж; для бару береться година з найдовшим рейнджем серед останніх 24 год.
Ознаки:
    rng2_dur      тривалість рейнджу, год (0 — рейнджу ≥ 12 год не було)
    rng2_width    ширина коридору, лог
    rng2_width_x  ширина ÷ медіана денного діапазону монети
    rng2_brk      (ціна − верх) / (верх − низ) — пробій у ширинах рейнджу (від'ємне — ще всередині / нижче)
    rng2_age      годин від кінця рейнджу (0 — ціна ще в коридорі)
Варіанти: range (ознаки робочої моделі + рейндж v1), range2 (робочі + v2 замість v1), range12 (обидва). Перевірка — як
experiments/range_features.py: leave-ticker-out на розмітці і бектест 6 міс. на чесному списку.
→ reports/range_v2.md, reports/range_v2_trades.csv

    python experiments/range_v2.py
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

LENS = (12, 24, 48, 72, 96, 144, 192, 288, 384, 504)
RANGE2 = ["rng2_dur", "rng2_width", "rng2_width_x", "rng2_brk", "rng2_age"]
BASE = MODEL_FEATURES_15M + R.MOVE
VARIANTS = {"range": BASE + R.RANGE, "range2": BASE + RANGE2, "range12": BASE + R.RANGE + RANGE2}
OUT = os.path.join("reports", "range_v2.md")
BAR = pd.Timedelta(minutes=15)


def range2_features(h: pd.DataFrame) -> pd.DataFrame:
    """15m-ряд (high, low, price) → ознаки рейнджу v2 на кожному барі (лише закриті години до закриття бару)."""
    out = pd.DataFrame(np.nan, index=h.index, columns=RANGE2)
    px = h["price"].where(h["price"] > 0)
    c = px.resample("1h").last()
    c.index = c.index + pd.Timedelta(hours=1)                          # кінець години
    dr = (h["high"].resample("1D").max() / h["low"].resample("1D").min() - 1)
    dr.index = dr.index + pd.Timedelta(days=1)
    norm = dr.rolling(30, min_periods=7).median()
    band = np.maximum(0.04, 1.5 * norm.reindex(c.index, method="ffill"))
    n = len(c)
    if n < 48:
        return out
    dur = np.zeros(n)
    top = np.full(n, np.nan)
    bot = np.full(n, np.nan)
    for L in LENS:                                                      # довші перезаписують коротші, якщо теж рейндж
        hi = c.rolling(L, min_periods=int(L * 0.9)).max()
        lo = c.rolling(L, min_periods=int(L * 0.9)).min()
        ok = ((hi / lo - 1) <= band).to_numpy()
        dur = np.where(ok, L, dur)
        top = np.where(ok, hi.to_numpy(), top)
        bot = np.where(ok, lo.to_numpy(), bot)
    # для кожної години — найдовший рейндж серед останніх 24 год (за рівності — найсвіжіший)
    jstar = np.zeros(n, dtype=int)
    for k in range(n):
        a = max(0, k - 23)
        seg = dur[a:k + 1]
        jstar[k] = a + (len(seg) - 1 - int(np.argmax(seg[::-1])))
    nrm = norm.reindex(c.index, method="ffill").to_numpy()
    # бар → остання закрита година на момент закриття бару
    k_of_bar = np.searchsorted(c.index.values, (h.index + BAR).values, side="right") - 1
    valid = k_of_bar >= 0
    kb = np.clip(k_of_bar, 0, n - 1)
    j = jstar[kb]
    d_ = dur[j]
    t_, b_ = top[j], bot[j]
    p = px.to_numpy()
    has = valid & (d_ > 0)
    out["rng2_dur"] = np.where(valid, d_, np.nan)
    out["rng2_width"] = np.where(has, np.log(t_ / b_), np.nan)
    out["rng2_width_x"] = np.where(has, (t_ / b_ - 1) / nrm[j], np.nan)
    out["rng2_brk"] = np.where(has, (p - t_) / (t_ - b_), np.nan)
    out["rng2_age"] = np.where(has, (kb - j).astype(float), np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Рейндж v2 (тривалість коридору по закриттях, пробій у ширинах) проти рейнджу v1 у 15m-моделі").parse_args()
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
        nf = I.new_features(a, idx).join(R.range_features(a, a["oi_coin_log_idx"])).join(range2_features(a))
        parts.append(nf.reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts)).sort_values(["base_asset", "ts"]).reset_index(drop=True)
    y = d["label"].astype(int)
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів; рейндж v2 ≥ 12 год є в {(d['rng2_dur'] > 0).mean():.0%} барів, медіана тривалості "
          f"{d.loc[d['rng2_dur'] > 0, 'rng2_dur'].median():.0f} год", flush=True)
    models, lto, imp = {}, {}, None
    for name, cols in VARIANTS.items():
        p = train.oof_proba(train.make_lightgbm, d[cols], y, d["base_asset"])
        thr = train.best_f1_threshold(y.to_numpy(), p)
        lto[name] = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        mo = train.make_lightgbm().fit(d[cols], y)
        models[name] = (mo, thr, cols)
        if name == "range12":
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
        nf = I.new_features(h, idx).join(R.range_features(h, R.archive_oi_log(c, h.index))).join(range2_features(h))
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
    tr.to_csv(os.path.join("reports", "range_v2_trades.csv"), index=False)
    pf = I.pf
    names = list(VARIANTS)
    L = ["# Рейндж v2 проти рейнджу v1", "",
         "Усі варіанти свіжонавчені однаково (тікери навчання, лише Binance). range = робочі + рейндж v1, range2 = робочі + " + ", ".join(RANGE2)
         + " (замість v1), range12 = обидва.", "",
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
    L += ["", "Місяців кращих за range (v1): " + ", ".join(
        f"{v} усі {int((piv[v] > piv['range'] + 1e-9).sum())} / ≥ 20 млн {int((piv20[v] > piv20['range'] + 1e-9).sum())} з {len(piv)}"
        for v in names if v != "range"), ""]
    if imp is not None:
        L += ["## Важливість у варіанті range12 (gain)", "",
              f"Рейндж v2 разом {imp[RANGE2].sum():.1%} — " + ", ".join(f"{k} {imp[k]:.1%}" for k in RANGE2) + f"; рейндж v1 разом {imp[R.RANGE].sum():.1%}", "",
              "Топ-10: " + ", ".join(f"{k} {v:.1%}" for k, v in imp.sort_values(ascending=False).head(10).items()), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
