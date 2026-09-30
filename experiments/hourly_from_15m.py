"""
experiments/hourly_from_15m.py — чи можна годинний контекст (підтвердження Alert) рахувати з НАШИХ 15m-даних
замість годинного збору Coinalyze.

Навіщо: годинна частина циклу скринера — ~23 виклики Coinalyze на тікер (40/хв), на 18 тікерах ~12 хв.
Весь цей час Alert чекає. 15m-дані з бірж (Binance+Bybit) і 15m-ліквідації Coinalyze ми й так тягнемо щоцикла;
якщо зібрати з них годинні бари, годинна модель рахувалась би в тому ж 15m-циклі, о HH:02.

Порівнюємо на тих самих годинах і тікерах:
    A — як зараз: годинні ряди Coinalyze (6 бірж) → features → годинний LightGBM
    B — 15m-ряди з бази (Binance+Bybit + ліквідації Coinalyze) → зібрані в годину → ті самі features → той самий LightGBM
Метрики: кореляція ознак, кореляція P, і головне — збіг рішення «годинне підтвердження є» (P ≥ 0.75·поріг),
а також події Alert на 15m-сітці (15m-модель + підтвердження з A або з B) проти розмітки.

    python experiments/hourly_from_15m.py               # тікери з tickers.txt
    python experiments/hourly_from_15m.py --days 30     # лише останні N днів
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from features import FEATURES, INTERVAL_15M, build_15m, build_from_raw, load_pumps, load_raw
from fetch import read_tickers_file
from tiers import tier_frame


def to_hourly(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """15m-ряди кожного контракту → години (лише повні години з 4 барів; ts — відкриття години)."""
    keys = ["base_asset", "market", "exchange", "quote_asset"]
    out = {}

    def roll(df, agg, extra_keys=()):
        if df.empty:
            return df
        d = df.assign(ts=df["ts"].dt.floor("h"))
        g = d.groupby(keys + list(extra_keys) + ["ts"], dropna=False)
        res = g.agg(**agg, n=("ts", "size")).reset_index()
        return res[res["n"] == 4].drop(columns="n")

    out["ohlcv"] = roll(raw["ohlcv"], dict(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
                                           v=("v", "sum"), bv=("bv", lambda x: x.sum(min_count=1))), ["volume_denominated"])
    out["oi"] = roll(raw["oi"], dict(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last")))
    out["funding"] = roll(raw["funding"], dict(c=("c", "last")))
    out["ls"] = roll(raw["ls"], dict(long_pct=("long_pct", "last")))
    out["liq"] = roll(raw["liq"], dict(long_liq=("long_liq", "sum"), short_liq=("short_liq", "sum")))
    if "liq_coverage" in raw:
        out["liq_coverage"] = raw["liq_coverage"]
    for k, v in out.items():
        if k != "liq_coverage" and not v.empty:
            v.sort_values(keys[:2] + ["ts"], inplace=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Годинний контекст із власних 15m-даних проти Coinalyze 1h")
    parser.add_argument("--days", type=int, default=60, help="скільки останніх днів порівнювати (за замовчуванням 60)")
    args = parser.parse_args()
    load_dotenv(".env")
    bases = [t.upper() for t in read_tickers_file("tickers.txt")]
    m1 = joblib.load(train.model_path("lightgbm"))
    m15 = joblib.load(train.model_path(train.M15_NAME))
    share = m15.get("confirm", {}).get("share", 0.75)
    need = share * m1["threshold"]
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days + 8)   # +8 днів на вікна ознак

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        fa = build_from_raw(load_raw(conn, bases, since=since))                       # A: Coinalyze 1h
        fb = build_from_raw(to_hourly(load_raw(conn, bases, INTERVAL_15M, since=since)))  # B: наші 15m → 1h
        d15 = build_15m(conn, bases, since=since)

    cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)
    m = fa.merge(fb, on=["base_asset", "ts"], suffixes=("_a", "_b"))
    m = m[m["ts"] >= cut]
    print(f"Спільних годин: {len(m)} на {m['base_asset'].nunique()} тікерах (A: {len(fa)}, B: {len(fb)} рядків)")

    print("\nКореляція ознак A↔B (Спірмен), найгірші:")
    corr = {f: m[f + "_a"].corr(m[f + "_b"], method="spearman") for f in FEATURES}
    for f, c in sorted(corr.items(), key=lambda x: (x[1] if pd.notna(x[1]) else -9))[:10]:
        print(f"  {f:<24} {c:.3f}")

    pa = m1["model"].predict_proba(m[[f + "_a" for f in FEATURES]].set_axis(FEATURES, axis=1))[:, 1]
    pb = m1["model"].predict_proba(m[[f + "_b" for f in FEATURES]].set_axis(FEATURES, axis=1))[:, 1]
    ca, cb = pa >= need, pb >= need
    print(f"\nP годинної моделі: кореляція {np.corrcoef(pa, pb)[0, 1]:.3f}, медіана |A−B| {np.median(np.abs(pa - pb)):.3f}")
    print(f"Підтвердження (P ≥ {need:.2f}): A {ca.mean():.1%} годин, B {cb.mean():.1%}; збіг рішень {np.mean(ca == cb):.1%}")
    print(f"  лише A: {np.sum(ca & ~cb)} год, лише B: {np.sum(~ca & cb)} год, обидва: {np.sum(ca & cb)} год")

    # Alert на 15m-сітці з контекстом A і з контекстом B — проти розмітки (in-sample для тікерів навчання,
    # але порівнюємо лише джерело контексту, модель та сама)
    s15 = d15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(d15[train.FEATURES_15M])[:, 1],
                                          threshold=m15["threshold"])
    s15 = s15[s15["ts"] >= cut]
    print("\nAlert на 15m-сітці проти розмітки:")
    for name, f, p in (("A Coinalyze 1h", m[["base_asset", "ts"]], pa), ("B власні 15m→1h", m[["base_asset", "ts"]], pb)):
        s1 = f.assign(probability=p, threshold=m1["threshold"])
        t = tier_frame(s15, s1, share)
        ev = train.event_metrics(t[["base_asset", "ts"]], (t["tier"] == "alert").to_numpy(), pumps, train.BAR_15M)
        print(f"  {name:<18} спіймано {ev['pumps_caught']} · фальстарти {ev['false_starts_triggered']} · "
              f"хибних / 30 д {ev['false_alarms_per_30d']:.1f} · затримка {ev['median_delay_h'] * 60 if ev['median_delay_h'] is not None else float('nan'):.0f} хв")


if __name__ == "__main__":
    main()
