"""
experiments/bb_features.py — смуги Боллінджера (20, 2σ) як ознаки 15m-моделі поверх рейнджу (2026-10-09, власник: «щось типу
bollinger bands, щоб визначати тренд»). Нове для моделі — контекст старшого таймфрейму (у моделі немає нічого довшого за добу);
решта — перевірка, чи дублює вже наявне (волатильність, рух у σ, рейндж).

Ознаки (зафіксовано ДО запуску; старші таймфрейми — лише ЗАКРИТІ свічки на момент закриття 15m-бару, ціна — поточна):
    bb_pctb_4h     %B поточної ціни в смузі 4h: (ціна − нижня) / (верхня − нижня)
    bb_slope_4h    нахил середньої лінії 4h за 5 свічок, у σ смуги
    bb_pctb_1d     те саме для денних свічок
    bb_slope_1d    нахил денної середньої за 5 днів, у σ
    bb_walk_15m    скільки з останніх 8 закриттів 15m над верхньою смугою 15m («прогулянка по смузі»)
    bb_width_x_1h  ширина смуги 1h (4σ / середня) ÷ медіана за 30 днів (стиснення / розширення)
Варіанти: range (= ознаки робочої моделі + рейндж) і range_bb. Перевірка — як experiments/range_features.py.
→ reports/bb_features.md, reports/bb_features_trades.csv

    python experiments/bb_features.py
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

BB = ["bb_pctb_4h", "bb_slope_4h", "bb_pctb_1d", "bb_slope_1d", "bb_walk_15m", "bb_width_x_1h"]
RANGE_SET = MODEL_FEATURES_15M + R.MOVE + R.RANGE
VARIANTS = {"range": RANGE_SET, "range_bb": RANGE_SET + BB}
OUT = os.path.join("reports", "bb_features.md")
BAR = pd.Timedelta(minutes=15)


def _closed(price: pd.Series, rule: str, td: pd.Timedelta) -> pd.Series:
    """Закриття свічок старшого ТФ з індексом = час закриття свічки (лише закриті потрапляють у бари після нього)."""
    c = price.resample(rule).last()
    c.index = c.index + td
    return c


def bb_features(h: pd.DataFrame) -> pd.DataFrame:
    px = h["price"].where(h["price"] > 0)
    at = h.index + BAR                                   # момент закриття 15m-бару
    f = pd.DataFrame(index=h.index)
    for rule, td, tag in (("4h", pd.Timedelta(hours=4), "4h"), ("1D", pd.Timedelta(days=1), "1d")):
        c = _closed(px, rule, td)
        sma, sd = c.rolling(20, min_periods=20).mean(), c.rolling(20, min_periods=20).std()
        slope = (sma - sma.shift(5)) / sd
        m = pd.DataFrame({"sma": sma, "sd": sd, "slope": slope}).reindex(at, method="ffill").set_axis(h.index)
        f[f"bb_pctb_{tag}"] = (px - (m["sma"] - 2 * m["sd"])) / (4 * m["sd"])
        f[f"bb_slope_{tag}"] = m["slope"]
    sma15, sd15 = px.rolling(20).mean(), px.rolling(20).std()
    f["bb_walk_15m"] = (px > sma15 + 2 * sd15).astype(float).rolling(8).sum()
    c1 = _closed(px, "1h", pd.Timedelta(hours=1))
    w = 4 * c1.rolling(20).std() / c1.rolling(20).mean()
    wx = w / w.shift(1).rolling(720, min_periods=240).median()
    f["bb_width_x_1h"] = wx.reindex(at, method="ffill").set_axis(h.index)
    return f.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Смуги Боллінджера (тренд 4h / 1d, прогулянка по смузі, стиснення) поверх ознак рейнджу").parse_args()
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
        nf = I.new_features(a, idx).join(R.range_features(a, a["oi_coin_log_idx"])).join(bb_features(a))
        parts.append(nf.reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts)).sort_values(["base_asset", "ts"]).reset_index(drop=True)
    y = d["label"].astype(int)
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів; тренд 4h є в {d['bb_slope_4h'].notna().mean():.0%}, денний — в "
          f"{d['bb_slope_1d'].notna().mean():.0%} барів", flush=True)
    models, lto, imp = {}, {}, None
    for name, cols in VARIANTS.items():
        p = train.oof_proba(train.make_lightgbm, d[cols], y, d["base_asset"])
        thr = train.best_f1_threshold(y.to_numpy(), p)
        lto[name] = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        mo = train.make_lightgbm().fit(d[cols], y)
        models[name] = (mo, thr, cols)
        if name == "range_bb":
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
        nf = I.new_features(h, idx).join(R.range_features(h, R.archive_oi_log(c, h.index))).join(bb_features(h))
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
    tr.to_csv(os.path.join("reports", "bb_features_trades.csv"), index=False)
    pf, names = I.pf, list(VARIANTS)
    L = ["# Смуги Боллінджера поверх ознак рейнджу", "",
         "range = ознаки робочої моделі + рейндж; range_bb = + " + ", ".join(BB) + ". Обидві свіжонавчені однаково.", "",
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
    L += ["", "## По місяцях (PF; у дужках ≥ 20 млн)", "", "| місяць | range | range_bb |", "|---|---|---|"]
    for mth in piv.index:
        L.append(f"| {mth} | {piv.at[mth, 'range']:.2f} ({piv20.at[mth, 'range']:.2f}) | {piv.at[mth, 'range_bb']:.2f} ({piv20.at[mth, 'range_bb']:.2f}) |")
    L += ["", f"Місяців, де range_bb кращий: усі {int((piv['range_bb'] > piv['range'] + 1e-9).sum())}, ≥ 20 млн "
          f"{int((piv20['range_bb'] > piv20['range'] + 1e-9).sum())} з {len(piv)}", ""]
    if imp is not None:
        L += [f"Важливість Боллінджера разом {imp[BB].sum():.1%}: " + ", ".join(f"{k} {imp[k]:.1%}" for k in BB), "",
              "Топ-10: " + ", ".join(f"{k} {v:.1%}" for k, v in imp.sort_values(ascending=False).head(10).items()), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
