"""
Експеримент: два рівні сигналу — 15m «Кандидат» і підтвердження годинною моделлю («Alert»).

15m-модель бачить імпульс на ~70 хв раніше за годинну, але спрацьовує на ~⅔ фальстартів (experiments/m15_model.py).
Перевіряємо, чи годинний контекст відсіює фальстарти і скільки хвилин це коштує.

Обидві моделі — LightGBM, leave-ticker-out з вкладеним порогом (out-of-fold: кожен тікер оцінює модель, що його
не бачила). Оцінки нормовано на поріг: s ≥ 1 — сигнал. До 15m-бару береться оцінка ОСТАННЬОЇ ЗАКРИТОЇ години
(15m-бар 13:30–13:45 → годинний бар 12:00–13:00), тобто лише те, що було відомо на момент закриття 15m-бару.

Правила на 15m-барах:
    15m                    s15 ≥ 1                         — «Кандидат» без підтвердження
    15m & 1h ≥ c           s15 ≥ 1 і s1h ≥ c (c = 0.5, 0.75, 1) — «Alert»: годинна модель теж «гріється»
    15m & 1h ≥ c за 4 год  те саме, але s1h ≥ c хоч раз за останні 4 закриті години (підтвердження «пам'ятає»)
    1h                     s1h ≥ 1 — лише годинна модель (на 15m-сітці, для порівняння)

    python experiments/two_tier.py      → reports/experiment_two_tier.md
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
from features import FEATURES, build_from_raw, load_pumps, load_raw, query_df  # noqa: E402
from fetch import TICKERS_FILE, read_tickers_file  # noqa: E402
from m15_model import SHORT, SHORT_LIQ, short_features  # noqa: E402
from timeframe import events  # noqa: E402

REPORT = os.path.join(ROOT, "reports", "experiment_two_tier.md")
H = pd.Timedelta(hours=1)
BAR = pd.Timedelta(minutes=15)
M15_FEATURES = FEATURES + SHORT + SHORT_LIQ


def main() -> None:
    argparse.ArgumentParser(description="Експеримент: 15m-кандидат + підтвердження годинною моделлю "
                                        "→ reports/experiment_two_tier.md").parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    bases = [t.upper() for t in read_tickers_file(os.path.join(ROOT, TICKERS_FILE))]
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        m15 = build_from_raw(load_raw(conn, bases, "15min"), pumps, bars_per_hour=4, extra=short_features)
        h1 = query_df(conn, "SELECT * FROM features WHERE base_asset = ANY(%s) AND reviewed", (bases,))

    # вікно — як у m15_model.py: перетин періодів, без свіжих нерозмічених годин
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
    print(f"Вікно: {lo.min():%m-%d}…{hi.max():%m-%d}, {len(lo)} тікерів, {n_pump} пампів")

    # out-of-fold оцінки обох моделей, нормовані на поріг
    print("15m LightGBM…")
    r15 = train.leave_ticker_out("lightgbm", m15[M15_FEATURES], m15["label"].astype(int), m15["base_asset"])
    print("1h LightGBM…")
    r1 = train.leave_ticker_out("lightgbm", h1[FEATURES], h1["label"].astype(int), h1["base_asset"])
    s15 = (r15["score"] / r15["threshold"]).to_numpy()
    hour = h1[["base_asset", "ts"]].assign(s1h=(r1["score"] / r1["threshold"]).to_numpy())
    # «пам'ять» підтвердження: максимум годинної оцінки за останні 4 закриті години
    hour["s1h_4h"] = hour.groupby("base_asset")["s1h"].transform(lambda s: s.rolling(4, min_periods=1).max())

    # останній закритий годинний бар на момент закриття 15m-бару
    key = (m15["ts"] + BAR - H).dt.floor("h")
    ctx = m15[["base_asset"]].assign(ts=key).merge(hour, on=["base_asset", "ts"], how="left")
    s1h, s1h_4h = ctx["s1h"].fillna(0).to_numpy(), ctx["s1h_4h"].fillna(0).to_numpy()

    rules = {"15m (candidate)": s15 >= 1}
    for c in (0.5, 0.75, 1.0):
        rules[f"15m & 1h ≥ {c:g}"] = (s15 >= 1) & (s1h >= c)
    for c in (0.5, 1.0):
        rules[f"15m & 1h ≥ {c:g} within 4 h"] = (s15 >= 1) & (s1h_4h >= c)
    rules["1h only"] = s1h >= 1

    fine_price = m15[["base_asset", "ts", "price"]]
    rows = {name: events(m15, sig, BAR, ev_pumps, fine_price) for name, sig in rules.items()}
    out = pd.DataFrame(rows).T
    print(out.to_string())

    # скільки чекати від «Кандидата» до «Alert» на спійманих пампах
    lag = []
    for p in ev_pumps[ev_pumps["pattern_type"] == "pump"].itertuples():
        own = (m15["base_asset"] == p.base_asset) & (m15["ts"] + BAR > p.start_ts) & (m15["ts"] + BAR <= p.end_ts)
        c_ts, a_ts = m15.loc[own & rules["15m (candidate)"], "ts"], m15.loc[own & rules["15m & 1h ≥ 0.5 within 4 h"], "ts"]
        if len(c_ts) and len(a_ts):
            lag.append((a_ts.min() - c_ts.min()) / pd.Timedelta(minutes=1))
    lag_txt = (f"Median wait from Candidate to Alert (1h ≥ 0.5 within 4 h) on pumps caught by both: "
               f"{pd.Series(lag).median():.0f} min ({len(lag)} pumps).") if lag else ""
    print(lag_txt)

    lines = [
        "# Experiment: two-tier signal — 15m candidate, hourly confirmation", "",
        f"{len(lo)} tickers, {lo.min():%Y-%m-%d} – {hi.max():%Y-%m-%d}, {n_pump} labeled pumps (impulse labels). "
        "Out-of-fold LightGBM scores (leave-ticker-out, nested threshold), normalised by threshold. Each 15m bar gets "
        "the hourly score of the last hourly bar closed by the 15m bar's close, so no look-ahead.", "",
        "- **15m (candidate)**: 15m model alone (liquidations + short features)",
        "- **15m & 1h ≥ c**: 15m signal and the hourly score is at least c × its threshold",
        "- **… within 4 h**: the hourly condition held at least once in the last 4 closed hours",
        "- **1h only**: hourly model alone, on the 15m grid", "",
        train.md_table(out), "", lag_txt, "",
    ]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nЗвіт: {REPORT}")


if __name__ == "__main__":
    main()
