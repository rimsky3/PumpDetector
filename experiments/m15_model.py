"""
Експеримент: 15m-модель — чи зменшують хибні спрацювання ліквідації і короткі ознаки.

Дані — з бази, interval '15min': свічки, OI, funding, L/S з Binance і Bybit (fetch_exchange.py) і ліквідації
з Coinalyze з покриттям (до covered_from ринку — «невідомо», не нуль). Розмітка — імпульси з pumps.csv.

Варіанти ознак на 15m-барах (вікна за часом ті самі, що на 1h: «24h» — доба):
    base         FEATURES без ліквідацій (як у experiments/timeframe.py)
    +liq         усі FEATURES, з 15m-ліквідаціями
    +short       base + короткі ознаки (бар = 15 хв): зміни за 1–2 бари, прискорення за годину, сплеск обсягу
    +liq+short   усе разом, плюс ліквідації за останню годину
Для порівняння — поточна годинна модель (1h Coinalyze, FEATURES) у тому самому вікні.
Leave-ticker-out з вкладеним порогом, як у train.py; метрики подій — як у experiments/timeframe.py.

    python experiments/m15_model.py      → reports/experiment_m15_model.md
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402
from features import FEATURES, SHORT_FEATURES, build_from_raw, load_pumps, load_raw, query_df, short_features  # noqa: E402
from fetch import TICKERS_FILE, read_tickers_file  # noqa: E402
from timeframe import events  # noqa: E402

REPORT = os.path.join(ROOT, "reports", "experiment_m15_model.md")
H = pd.Timedelta(hours=1)
BAR = pd.Timedelta(minutes=15)
LIQ = [f for f in FEATURES if f.startswith("liq_")]
BASE = [f for f in FEATURES if f not in LIQ]
SHORT_LIQ = ["liq_long_1h_oi", "liq_short_1h_oi"]
SHORT = [f for f in SHORT_FEATURES if f not in SHORT_LIQ]
VARIANTS = {"base": BASE, "+liq": FEATURES, "+short": BASE + SHORT, "+liq+short": FEATURES + SHORT + SHORT_LIQ}


def main() -> None:
    argparse.ArgumentParser(description="Експеримент: 15m-модель з ліквідаціями і короткими ознаками "
                                        "→ reports/experiment_m15_model.md").parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    bases = [t.upper() for t in read_tickers_file(os.path.join(ROOT, TICKERS_FILE))]
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        m15 = build_from_raw(load_raw(conn, bases, "15min"), pumps, bars_per_hour=4, extra=short_features)
        h1 = query_df(conn, "SELECT * FROM features WHERE base_asset = ANY(%s) AND reviewed", (bases,))

    # однакове вікно: перетин періодів 1h і 15m кожного тікера; свіжі нерозмічені години — як у train.py
    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(hours=train.FRESH_HOURS)
    span = pd.concat([h1.groupby("base_asset")["ts"].agg(["min", "max"]),
                      m15.groupby("base_asset")["ts"].agg(["min", "max"])], axis=1, keys=["a", "b"]).dropna()
    lo = span[[("a", "min"), ("b", "min")]].max(axis=1).dt.ceil("h")
    hi = span[[("a", "max"), ("b", "max")]].min(axis=1)

    def window(d):
        d = d[d["base_asset"].isin(lo.index)]
        keep = (d["ts"] >= d["base_asset"].map(lo)) & (d["ts"] <= d["base_asset"].map(hi))
        keep &= (d["ts"] <= cutoff) | d["pattern_type"].notna()
        return d[keep].reset_index(drop=True)

    m15, h1 = window(m15), window(h1)
    ev_pumps = pumps[pumps["base_asset"].isin(lo.index) & (pumps["start_ts"] >= pumps["base_asset"].map(lo))
                     & (pumps["start_ts"] <= pumps["base_asset"].map(hi))]
    n_pump = int((ev_pumps["pattern_type"] == "pump").sum())
    fine_price = m15[["base_asset", "ts", "price"]]
    liq_known = m15["liq_long_4h_oi"].notna().mean()
    print(f"Вікно: {lo.min():%m-%d}…{hi.max():%m-%d}, {len(lo)} тікерів, {n_pump} пампів; рядків 1h {len(h1)}, 15m {len(m15)}; "
          f"ліквідації відомі на {liq_known:.0%} 15m-барів")

    rows, imp = {}, None
    runs = [("1h", h1, H, "1h (current)", FEATURES, ("rule", "logreg", "lightgbm"))]
    runs += [("15m", m15, BAR, v, cols, ("rule", "logreg", "lightgbm") if v == "base" else ("logreg", "lightgbm"))
             for v, cols in VARIANTS.items()]
    for tf, d, bar, name, cols, models in runs:
        y, groups = d["label"].astype(int), d["base_asset"]
        for m in models:
            print(f"  {name} / {m}")
            res = train.leave_ticker_out(m, d[cols], y, groups)
            rows[f"{m} · {name}"] = {**train.hour_metrics(y, res, m != "rule"),
                                     **events(d, res["signal"].to_numpy(), bar, ev_pumps, fine_price)}
    full = FEATURES + SHORT + SHORT_LIQ
    imp = train.feature_importance(m15[full], m15["label"].astype(int), ["lightgbm"]).head(15)
    out = pd.DataFrame(rows).T
    print(out.to_string())
    print(imp.to_string())

    lines = [
        "# Experiment: 15-minute model with liquidations and short features", "",
        f"{len(lo)} tickers from tickers.txt, {lo.min():%Y-%m-%d} – {hi.max():%Y-%m-%d}, {n_pump} labeled pumps "
        "(impulse labels on 5m/15m). 15m data from the database: Binance + Bybit candles, OI, funding, long/short, and "
        f"Coinalyze 15m liquidations, known on {liq_known:.0%} of bars (Coinalyze keeps a limited number of points; "
        "before a market's coverage starts, liquidations are missing, not zero). Leave-ticker-out with nested threshold.", "",
        f"- **1h (current)**: production features on hourly Coinalyze data, same window",
        f"- **base**: {len(BASE)} features without liquidations, 15m bars, same wall-clock windows",
        "- **+liq**: all production features, liquidations from 15m data",
        f"- **+short**: base + {', '.join(SHORT)}",
        f"- **+liq+short**: everything, plus {', '.join(SHORT_LIQ)}", "",
        "Event metrics as in `experiments/timeframe.py`: delay to the close of the first signal bar, share of the move "
        "already gone, peak / signal price − 1, false episodes (bars ≤ 1 h apart) per ticker-month.", "",
        train.md_table(out), "",
        "## LightGBM gain, all 15m features (fitted on all data)", "",
        train.md_table(imp), "",
    ]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nЗвіт: {REPORT}")


if __name__ == "__main__":
    main()
