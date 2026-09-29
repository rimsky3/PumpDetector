"""
Експеримент: агрегована volume delta на перпах і на споті як нові ознаки.

Volume delta = агресивні покупки − агресивні продажі (taker buy − taker sell) = 2·buy volume − volume.
Частка покупок у нас уже є (buy_ratio_1h/4h = bv/v; дельта/обсяг = 2·buy_ratio − 1 — та сама інформація),
тому тестуємо форми, яких ще немає:

    перп (усі біржі з Coinalyze, як решта ознак)
        cvd_24h           кумулятивна дельта за добу / обсяг за добу
        vd_1h_oi          дельта години / OI — агресія відносно відкритих позицій
        cvd_24h_oi        дельта за добу / OI
        vd_z_7d           дельта години відносно тижневої норми (z-score)
    спот (Binance spot USDT — єдине джерело спотового taker buy для дрібних монет; не для всіх тікерів)
        spot_buy_ratio_4h частка покупок на споті за 4 год
        spot_cvd_24h      кумулятивна спот-дельта за добу / спот-обсяг
        spot_vd_z_7d      спот-дельта відносно тижневої норми
        spot_perp_vol_24h спот-обсяг / перп-обсяг за добу — памп тягне спот чи ф'ючерси
        spot_share_z_7d   те саме відносно тижневої норми

Варіанти: base (поточні FEATURES), +perp, +spot, +both. Leave-ticker-out з вкладеним порогом, як train.py.

    python experiments/volume_delta.py            → reports/experiment_volume_delta.md
    python experiments/volume_delta.py --refresh  # завантажити спот з Binance заново
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402
from exchange_direct import CACHE, DAYS  # noqa: E402
from exchanges import MS_H, get, to_ts  # noqa: E402
from features import BASE_WINDOW, FEATURES, SHORT_WINDOW, aggregate, load_pumps, load_raw, query_df, zscore  # noqa: E402

PERP = ["cvd_24h", "vd_1h_oi", "cvd_24h_oi", "vd_z_7d"]
SPOT = ["spot_buy_ratio_4h", "spot_cvd_24h", "spot_vd_z_7d", "spot_perp_vol_24h", "spot_share_z_7d"]
VARIANTS = {"base": FEATURES, "+perp": FEATURES + PERP, "+spot": FEATURES + SPOT, "+both": FEATURES + PERP + SPOT}
REPORT = os.path.join(ROOT, "reports", "experiment_volume_delta.md")


def binance_spot(base: str, refresh: bool) -> pd.DataFrame | None:
    """Годинні свічки Binance spot {base}USDT (з кешем). None — якщо споту на Binance немає."""
    path = os.path.join(CACHE, f"BinanceSpot_{base}USDT_klines.parquet")
    if os.path.exists(path) and not refresh:
        df = pd.read_parquet(path)
        return None if df.empty else df
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows, t = [], end - DAYS * 24 * MS_H
    try:
        while t < end:
            page = get("https://api.binance.com/api/v3/klines",
                       dict(symbol=f"{base}USDT", interval="1h", startTime=t, endTime=end, limit=1000))
            if not page:
                break
            rows += page
            t = page[-1][0] + MS_H
    except Exception:           # 400 Invalid symbol — пари на споті немає
        rows = []
    # [open_time, o, h, l, c, volume (монети), close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ...]
    df = pd.DataFrame({"ts": [to_ts(r[0]) for r in rows], "v": [float(r[7]) for r in rows],
                       "bv": [float(r[10]) for r in rows]})
    os.makedirs(CACHE, exist_ok=True)
    df.to_parquet(path, index=False)
    return None if df.empty else df


def delta_features(h: pd.DataFrame, spot: pd.DataFrame | None) -> pd.DataFrame:
    """Ті самі правила, що в features.py: лише дані до години t включно, відносно власної історії."""
    f = pd.DataFrame(index=h.index)
    vd = 2 * h["buy_volume"] - h["volume"]
    f["cvd_24h"] = vd.rolling(SHORT_WINDOW).sum() / h["volume"].rolling(SHORT_WINDOW).sum()
    f["vd_1h_oi"] = vd / h["oi"]
    f["cvd_24h_oi"] = vd.rolling(SHORT_WINDOW).sum() / h["oi"]
    f["vd_z_7d"] = zscore(vd, BASE_WINDOW)
    if spot is None:
        for c in SPOT:
            f[c] = np.nan
    else:
        s = spot.set_index("ts").reindex(h.index)
        s_v, s_bv = s["v"].fillna(0), s["bv"].fillna(0)      # година без угод на споті — нуль
        s_vd = 2 * s_bv - s_v
        f["spot_buy_ratio_4h"] = s_bv.rolling(4).sum() / s_v.rolling(4).sum()
        f["spot_cvd_24h"] = s_vd.rolling(SHORT_WINDOW).sum() / s_v.rolling(SHORT_WINDOW).sum()
        f["spot_vd_z_7d"] = zscore(s_vd, BASE_WINDOW)
        share = s_v.rolling(SHORT_WINDOW).sum() / h["volume"].rolling(SHORT_WINDOW).sum()
        f["spot_perp_vol_24h"] = share
        f["spot_share_z_7d"] = zscore(np.log(share.where(share > 0)), BASE_WINDOW)
    return f.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description="Експеримент: volume delta (перп і спот) як ознаки → reports/experiment_volume_delta.md")
    ap.add_argument("--refresh", action="store_true", help="завантажити спот Binance заново, ігноруючи кеш")
    args = ap.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        df = query_df(conn, "SELECT * FROM features ORDER BY base_asset, ts")
        bases = sorted(df["base_asset"].unique())
        raw = load_raw(conn, bases)
        pumps = load_pumps(conn)

    extra, has_spot = [], {}
    for b in bases:
        spot = binance_spot(b, args.refresh)
        has_spot[b] = spot is not None
        extra.append(delta_features(aggregate(raw, b), spot).assign(base_asset=b).reset_index())
    df = df.merge(pd.concat(extra, ignore_index=True), on=["base_asset", "ts"], how="left")
    print("Binance spot:", ", ".join(f"{b}{'' if s else ' —'}" for b, s in has_spot.items()))

    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(hours=train.FRESH_HOURS)
    df = df[(df["ts"] <= cutoff) | df["pattern_type"].notna()].reset_index(drop=True)
    y, groups = df["label"].astype(int), df["base_asset"]

    rows, gains = {}, {}
    for v, cols in VARIANTS.items():
        for m in ("logreg", "lightgbm"):
            print(f"{v} / {m}")
            res = train.leave_ticker_out(m, df[cols], y, groups)
            rows[f"{m} · {v}"] = {**train.hour_metrics(y, res, True), **train.event_metrics(df, res["signal"], pumps)}
        if v == "+both":
            imp = train.feature_importance(df[cols], y, ["lightgbm"])
            gains = imp.loc[[c for c in PERP + SPOT if c in imp.index]]
    out = pd.DataFrame(rows).T[["precision", "recall", "f1", "pr_auc", "pumps_caught",
                               "false_starts_triggered", "false_alarms_per_30d"]]
    print(out.to_string())
    print(gains.to_string())

    spot_list = ", ".join(b for b, s in has_spot.items() if s)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# Experiment: aggregated volume delta (perpetuals and spot)\n\n"
                f"{len(df)} hourly bars, {len(bases)} tickers, {int(y.sum())} pump hours. Leave-ticker-out with nested "
                "threshold, as in train.py. Perp delta from Coinalyze buy volume across exchanges; spot delta from "
                f"Binance spot (available for {spot_list}; other tickers have these features empty).\n\n"
                f"- **base**: current features ({len(FEATURES)})\n"
                f"- **+perp**: {', '.join(PERP)}\n"
                f"- **+spot**: {', '.join(SPOT)}\n"
                "- **+both**: all of them\n\n"
                "## Summary\n\n" + train.md_table(out) +
                "\n\n## LightGBM gain of the new features (+both, fitted on all data)\n\n" + train.md_table(gains) + "\n")
    print(f"\nЗвіт: {REPORT}")


if __name__ == "__main__":
    main()
