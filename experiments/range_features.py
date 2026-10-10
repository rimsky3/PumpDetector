"""
experiments/range_features.py — ознаки рейнджу В 15m-МОДЕЛІ проти фальстартів (2026-10-09; продовження experiments/range_breakout.py:
у розмітці власника справжні імпульси стартують зі спокійнішого стану — ширина 96 год 24% проти 50% у фальстартів, Кауфман
0.115 проти 0.151, OI за 48 год 5% проти 9%; модель пропустила SAND 02.10 після рейнджу).

Варіанти (обидва свіжонавчені однаково, як experiments/idio_features.py: тікери навчання, лише Binance):
    base   — MODEL_FEATURES_15M + власний рух / Amihud / волатильність (= ознаки робочої моделі; код idio_features)
    range  — base + RANGE (зафіксовано ДО запуску; вікно 96 год закінчується за 4 год до бару — останні години = сам імпульс):
             rng_96h   log(хай / лоу) за вікно
             rng_x96   rng_96h ÷ медіана таких діапазонів за 30 днів до того (стиснення відносно власної норми)
             er_96     ефективність Кауфмана 15m-закриттів за вікно (мало = боковик)
             oi_flat48 |зміна log OI у монетах| за 48 год, що закінчуються за 4 год до бару
             brk_rng   log(ціна / хай вікна) — наскільки ціна вже вище (нижче) рейнджу
Перевірка 1 — leave-ticker-out на розмітці (поріг — найкращий F1 на OOF, однаково для обох): «Кандидат» (лише 15m) — спіймано
імпульсів, фальстартів, хибних на 30 днів. Перевірка 2 — бектест як idio_features: 6 міс., чесний список (станом на місяць),
Alert = 15m + робоча годинна, частковий вихід, стоп 2.48 ATR; PF загалом, ≥ 20 млн, по місяцях. OI монет бектесту — архів
Binance metrics (5 хв, знімок +5 хв), навчання — oi_coin_log_idx з бази (лише Binance).
→ reports/range_features.md, reports/range_features_trades.csv

    python experiments/range_features.py
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
import idio_features as I
import train
from features import INTERVAL_15M, MODEL_FEATURES_15M, aggregate, build_from_raw, load_pumps, load_raw, query_df, short_features
from fetch import excluded_tickers, read_tickers_file
from tiers import tier_frame

ARCH = os.path.join("data", "binance_archive", "metrics")
GAP, W, W48 = 16, 384, 192            # 15m-барів: відступ 4 год, вікно 96 год, вікно OI 48 год
RANGE = ["rng_96h", "rng_x96", "er_96", "oi_flat48", "brk_rng"]
MOVE = I.IDIO + I.AMI + I.VOL
VARIANTS = {"base": MODEL_FEATURES_15M + MOVE, "range": MODEL_FEATURES_15M + MOVE + RANGE}
OUT = os.path.join("reports", "range_features.md")
BAR = pd.Timedelta(minutes=15)


def range_features(h: pd.DataFrame, oi_log: pd.Series) -> pd.DataFrame:
    """15m-ряд (high, low, price) + log OI у монетах на барах → ознаки рейнджу (лише минуле)."""
    f = pd.DataFrame(index=h.index)
    hi = h["high"].shift(GAP).rolling(W, min_periods=W * 3 // 4).max()
    lo = h["low"].shift(GAP).rolling(W, min_periods=W * 3 // 4).min()
    f["rng_96h"] = np.log(hi / lo)
    ref = f["rng_96h"].shift(W).rolling(30 * 96, min_periods=10 * 96).median()
    f["rng_x96"] = f["rng_96h"] / ref
    c = np.log(h["price"].where(h["price"] > 0)).shift(GAP)
    f["er_96"] = (c - c.shift(W)).abs() / c.diff().abs().rolling(W, min_periods=W * 3 // 4).sum()
    oi = oi_log.reindex(h.index).shift(GAP)
    f["oi_flat48"] = (oi - oi.shift(W48)).abs()
    f["brk_rng"] = np.log(h["price"] / hi)
    return f.replace([np.inf, -np.inf], np.nan)


def archive_oi_log(coin: str, index: pd.DatetimeIndex) -> pd.Series:
    """log OI у монетах з архіву Binance на 15m-барах (значення бару — стан на його закриття)."""
    files = sorted(glob.glob(os.path.join(ARCH, f"{coin}USDT", "*.parquet")))
    if not files:
        return pd.Series(np.nan, index=index)
    m = pd.concat([pd.read_parquet(f, columns=["create_time", "sum_open_interest"]) for f in files], ignore_index=True)
    s = pd.Series(m["sum_open_interest"].astype(float).to_numpy(), index=pd.to_datetime(m["create_time"], utc=True) + pd.Timedelta(minutes=5))
    s = s[s > 0]
    s = s[~s.index.duplicated()].sort_index()
    v = s.reindex(index + BAR, method="ffill", tolerance=pd.Timedelta(minutes=10)).to_numpy()
    return pd.Series(np.log(v), index=index)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Ознаки рейнджу в 15m-моделі: фальстарти (LTO) і бектест 6 міс. на чесному списку").parse_args()
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
        nf = I.new_features(a, idx).join(range_features(a, a["oi_coin_log_idx"]))
        parts.append(nf.reindex(g["ts"]).set_axis(g.index))
    d = d.join(pd.concat(parts)).sort_values(["base_asset", "ts"]).reset_index(drop=True)
    y = d["label"].astype(int)
    trained = set(d["base_asset"])
    print(f"навчання: {len(d)} барів, {d['base_asset'].nunique()} тікерів; ознаки рейнджу є в {d['rng_x96'].notna().mean():.0%}, "
          f"OI-рейнджу в {d['oi_flat48'].notna().mean():.0%} барів", flush=True)
    models, lto, imp = {}, {}, None
    for name, cols in VARIANTS.items():
        p = train.oof_proba(train.make_lightgbm, d[cols], y, d["base_asset"])
        thr = train.best_f1_threshold(y.to_numpy(), p)
        lto[name] = train.event_metrics(d, p >= thr, pumps, train.BAR_15M)
        mo = train.make_lightgbm().fit(d[cols], y)
        models[name] = (mo, thr, cols)
        if name == "range":
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
        nf = I.new_features(h, idx).join(range_features(h, archive_oi_log(c, h.index)))
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
    tr.to_csv(os.path.join("reports", "range_features_trades.csv"), index=False)
    pf = I.pf
    L = ["# Ознаки рейнджу в 15m-моделі", "",
         "Обидва варіанти свіжонавчені однаково (тікери навчання, лише Binance). base = ознаки робочої моделі "
         "(MODEL_FEATURES_15M + власний рух / Amihud / волатильність), range = base + " + ", ".join(RANGE) + ".", "",
         "## Leave-ticker-out на розмітці (Кандидат = лише 15m, поріг — найкращий F1 на OOF)", "",
         "| варіант | спіймано імпульсів | затримка, год | фальстартів спрацювало | хибних / 30 дн |", "|---|---|---|---|---|"]
    for v, m in lto.items():
        L.append(f"| {v} | {m['pumps_caught']} | {m['median_delay_h']:.2f} | {m['false_starts_triggered']} | {m['false_alarms_per_30d']:.1f} |")
    L += ["", f"## Бектест: {tr['ticker'].nunique()} монет чесного списку, {I.START} – {I.END}, Alert, частковий вихід, стоп {I.K} ATR", "",
          "| варіант | угод | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн |", "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        L.append(f"| {v} | {len(r)} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | {pf(r.iloc[:-3]):.2f} | "
                 f"{pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[list(VARIANTS)]
    piv20 = tr[tr["band"] == "≥ 20 млн"].groupby(["month", "variant"])["ret"].apply(pf).unstack()[list(VARIANTS)]
    L += ["", "## По місяцях (PF усі / ≥ 20 млн)", "", "| місяць | base | range | base ≥ 20 | range ≥ 20 |", "|---|---|---|---|---|"]
    for mth in piv.index:
        L.append(f"| {mth} | {piv.at[mth, 'base']:.2f} | {piv.at[mth, 'range']:.2f} | {piv20.at[mth, 'base']:.2f} | {piv20.at[mth, 'range']:.2f} |")
    L += ["", f"Місяців, де range кращий за base: усі {int((piv['range'] > piv['base'] + 1e-9).sum())} з {len(piv)}, "
          f"≥ 20 млн {int((piv20['range'] > piv20['base'] + 1e-9).sum())} з {len(piv20)}", ""]
    if imp is not None:
        L += ["## Важливість у варіанті range (gain)", "",
              "Ознаки рейнджу разом: " + f"{imp[RANGE].sum():.1%} — " + ", ".join(f"{k} {imp[k]:.1%}" for k in RANGE), "",
              "Топ-10: " + ", ".join(f"{k} {v:.1%}" for k, v in imp.sort_values(ascending=False).head(10).items()), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
