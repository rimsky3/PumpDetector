"""
Експеримент: OI у монетах проти OI у USD.

Coinalyze переводить OI у USD за ціною закриття години, тож USD-OI змішує приплив позицій з рухом ціни.
Гіпотеза: ознаки на OI у монетах (чистий приплив позицій) кращі.

Варіанти на тих самих даних і розмітці, leave-ticker-out з вкладеним порогом (як train.py):
    usd    — поточні FEATURES
    coins  — OI-ознаки (зміни, z-score, oi_minus_ret) замінено на монетні
    both   — поточні + монетні

    python experiments/oi_coins.py        → reports/experiment_oi_coins.md
"""

import argparse
import os
import sys

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import train  # noqa: E402
from features import FEATURES, OI_COIN_FEATURES, load_pumps, query_df  # noqa: E402

USD_OI = ["oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_z_24h", "oi_z_7d", "oi_minus_ret_24h"]
# у варіанті coins монетні ознаки підставляються під імена USD, щоб правило (oi_z_7d) теж працювало на монетах
COIN_AS_USD = {"oi_coin_chg_1h": "oi_chg_1h", "oi_coin_chg_4h": "oi_chg_4h", "oi_coin_chg_24h": "oi_chg_24h",
               "oi_coin_z_24h": "oi_z_24h", "oi_coin_z_7d": "oi_z_7d"}
REPORT = os.path.join(ROOT, "reports", "experiment_oi_coins.md")


def variant_X(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "usd":
        return df[FEATURES]
    if name == "coins":
        # oi_minus_ret_24h у монетах втрачає сенс: монетний OI уже не містить ціни
        cols = [c for c in FEATURES if c not in USD_OI] + OI_COIN_FEATURES
        return df[cols].rename(columns=COIN_AS_USD)
    return df[FEATURES + OI_COIN_FEATURES]


def main() -> None:
    argparse.ArgumentParser(description="Експеримент: ознаки на OI у монетах проти OI у USD → reports/experiment_oi_coins.md").parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        df = query_df(conn, "SELECT * FROM features ORDER BY base_asset, ts")
        pumps = load_pumps(conn)
    y, groups = df["label"].astype(int), df["base_asset"]
    tickers = sorted(groups.unique())

    summary, f1_by_ticker = {}, {}
    for v in ("usd", "coins", "both"):
        X = variant_X(df, v)
        for m in ("rule", "logreg", "lightgbm"):
            if m == "rule" and v == "both":
                continue  # правило бере лише oi_z_7d і обсяг — «both» для нього те саме, що usd
            print(f"{v} / {m}")
            res = train.leave_ticker_out(m, X, y, groups)
            key = f"{m} · {v}"
            summary[key] = {**train.hour_metrics(y, res, m != "rule"), **train.event_metrics(df, res["signal"], pumps)}
            f1_by_ticker[key] = {t: train.f1_score(y[groups == t], res["signal"][groups == t], zero_division=0) for t in tickers}

    s = pd.DataFrame(summary).T[["precision", "recall", "f1", "pr_auc", "pumps_caught", "false_starts_triggered", "false_alarms_per_30d"]]
    s.index.name = "model · variant"
    t = pd.DataFrame(f1_by_ticker).T
    t.index.name = "model · variant"
    print(s.to_string())
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("# Experiment: open interest in coins vs USD\n\n"
                f"{len(df)} hourly bars, {len(tickers)} tickers, {int(y.sum())} pump hours. "
                "Leave-ticker-out with nested threshold selection, as in train.py.\n\n"
                "- **usd** — current features (OI in USD as provided by Coinalyze)\n"
                "- **coins** — OI features (changes, z-scores) computed on OI in coins; `oi_minus_ret_24h` dropped\n"
                "- **both** — current features plus the coin-based ones\n\n"
                "## Summary\n\n" + train.md_table(s) + "\n\n## F1 by ticker\n\n" + train.md_table(t) + "\n")
    print(f"\nЗвіт: {REPORT}")


if __name__ == "__main__":
    main()
