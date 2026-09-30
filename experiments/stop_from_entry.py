"""
Експеримент: стоп від фактичного входу моделі, а не від розміченого старту пампу.

Зараз k (стоп = вхід − k·ATR) — 75-й перцентиль просідань від СТАРТУ розміченого пампу, в ATR на старті
(risk.pump_mae). Але 15m-контур заходить на першому Alert, у середньому через 25–45 хв після старту: ціна вже
вище, волатильність у кілька разів більша, а відкати всередині імпульсу глибші. Стоп з «тихого» старту вибиває
позиції на звичайних відкатах (SOON 27.09: відкат 1.20 ATR проти стопу 1.16, потім +20%).

Тут:
    1. out-of-fold оцінки 15m-моделі (leave-ticker-out, як у train.py) і годинного LightGBM (reports/oof_predictions.csv)
       → рівні сигналу, вхід = перший Alert усередині розміченого пампу;
    2. просідання від ціни входу до піку пампу в ATR(15m) на момент входу → перцентилі (новий k);
    3. симуляція угод на всіх out-of-fold Alert різних тікерів з різними k (без комісій) — який стоп дає найкращий результат.

    python experiments/stop_from_entry.py      → reports/experiment_stop.md
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
import train  # noqa: E402
from features import FEATURES_15M, INTERVAL_15M, aggregate, build_15m, in_pump, load_pumps, load_raw  # noqa: E402
from fetch import TICKERS_FILE, read_tickers_file  # noqa: E402
from risk import EXIT_BARS_15M, STOP_PERCENTILE, atr, position_state, stop_multiplier  # noqa: E402
from tiers import tier_frame  # noqa: E402

REPORT = os.path.join(ROOT, "reports", "experiment_stop.md")
BAR = pd.Timedelta(minutes=15)
KS = [1.16, 1.5, 2.0, 2.5, 3.0, 4.0]


def main() -> None:
    argparse.ArgumentParser(description="Експеримент: стоп від входу моделі → reports/experiment_stop.md").parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    reviewed = {t.upper() for t in read_tickers_file(os.path.join(ROOT, TICKERS_FILE))}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        bases = sorted(reviewed | set(pumps["base_asset"]))
        d = build_15m(conn, bases, pumps)
        raw = load_raw(conn, bases, INTERVAL_15M)
        k_old = stop_multiplier(conn, "15min").get("k")
    # та сама вибірка, що в train.py: переглянуті тікери — усі бари, інші — лише розмічені; без свіжих нерозмічених
    d["reviewed"] = d["base_asset"].isin(reviewed)
    cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(hours=train.FRESH_HOURS)
    d = d[(d["reviewed"] | d["pattern_type"].notna()) & ((d["ts"] <= cutoff) | d["pattern_type"].notna())].reset_index(drop=True)

    print(f"15m: {len(d)} барів, {d['base_asset'].nunique()} тікерів — out-of-fold LightGBM…")
    res = train.leave_ticker_out("lightgbm", d[FEATURES_15M], d["label"].astype(int), d["base_asset"])
    s15 = d[["base_asset", "ts"]].assign(probability=res["score"].to_numpy(), threshold=res["threshold"].to_numpy())
    oof = pd.read_csv(os.path.join(ROOT, "reports", "oof_predictions.csv"), parse_dates=["ts"])
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    s1 = oof[["base_asset", "ts"]].assign(probability=oof["lightgbm_score"], threshold=oof["lightgbm_threshold"])
    t = tier_frame(s15, s1, train.CONFIRM_SHARE)

    # 1-2. вхід моделі в кожному розміченому пампі і просідання від нього
    rows = []
    for base, g in t.groupby("base_asset"):
        h = aggregate(raw, base, "15min")
        if h.empty:
            continue
        a = atr(h)
        g = g.set_index("ts")
        for p in pumps[(pumps["base_asset"] == base) & (pumps["pattern_type"] == "pump")].itertuples():
            inside = g[in_pump(g.index, p.start_ts, p.end_ts, BAR) & (g["tier"] == "alert")]
            if inside.empty:
                continue
            e = inside.index[0]
            entry, atr_e = h.at[e, "price"], a.get(e)
            w = h[(h.index > e) & in_pump(h.index, p.start_ts, p.end_ts, BAR)]
            if pd.isna(entry) or pd.isna(atr_e) or not atr_e or w.empty:
                continue
            peak_ts = w["high"].idxmax()
            low = w.loc[:peak_ts, "low"].min()
            rows.append({"ticker": base, "start": p.start_ts, "entry": e + BAR, "delay_min": (e + BAR - p.start_ts) / pd.Timedelta(minutes=1),
                         "gone_pct": entry / h[h.index + BAR > p.start_ts]["open"].iloc[0] - 1,
                         "atr_pct": atr_e / entry, "mae_atr": max(0.0, (entry - low) / atr_e),
                         "peak_atr": (w["high"].max() - entry) / atr_e})
    m = pd.DataFrame(rows)
    pct = {f"p{q}": float(np.percentile(m["mae_atr"], q)) for q in (50, 75, 90)} if len(m) else {}
    print(m.describe().round(2).to_string())
    print("перцентилі просідання від входу, ATR:", {k: round(v, 2) for k, v in pct.items()})

    # 3. угоди на всіх out-of-fold Alert (вихід — Alert згас EXIT_BARS_15M барів) з різними k
    sims = []
    for k in KS + ([pct["p75"], pct["p90"]] if pct else []):
        res_k = []
        for base, g in t.groupby("base_asset"):
            h = aggregate(raw, base, "15min")
            if h.empty:
                continue
            pos = position_state(h, g.set_index("ts")["alert_score"], 1.0, k, EXIT_BARS_15M)
            for tr in pos.get("trades", []):
                own = pumps[(pumps["base_asset"] == base) & (pumps["start_ts"] < pd.Timestamp(tr["exit_ts"]) + BAR)
                            & (pumps["end_ts"] > pd.Timestamp(tr["entry_ts"]))]
                res_k.append((tr["result_pct"], tr["reason"], (own["pattern_type"] == "pump").any()))
        r = np.array([x[0] for x in res_k])
        in_p = np.array([x[2] for x in res_k])
        sims.append({"k": round(k, 2), "trades": len(r), "win_%": round((r > 0).mean() * 100, 1),
                     "stops_%": round(np.mean([x[1] == "stop" for x in res_k]) * 100, 1),
                     "avg_%": round(r.mean() * 100, 2), "median_%": round(np.median(r) * 100, 2),
                     "avg_in_pump_%": round(r[in_p].mean() * 100, 2) if in_p.any() else None,
                     "avg_outside_%": round(r[~in_p].mean() * 100, 2) if (~in_p).any() else None})
    sims = pd.DataFrame(sims).drop_duplicates("k").sort_values("k")
    print(sims.to_string(index=False))

    lines = [
        "# Experiment: stop measured from the model's entry", "",
        f"Out-of-fold 15m LightGBM (leave-ticker-out, {d['base_asset'].nunique()} tickers) with hourly confirmation "
        f"(out-of-fold, ≥ {train.CONFIRM_SHARE} of threshold). Entry = first Alert bar inside each labeled pump.", "",
        f"Current k = {k_old:.2f} ATR: {STOP_PERCENTILE}th percentile of drawdown from the **labeled start**, in ATR at the start.", "",
        f"## Drawdown from the model's entry ({len(m)} pumps with an Alert inside)", "",
        "| | median | 75th | 90th |", "|---|---|---|---|",
        f"| delay after start, min | {m['delay_min'].median():.0f} | {m['delay_min'].quantile(.75):.0f} | {m['delay_min'].quantile(.9):.0f} |" if len(m) else "",
        f"| move already gone at entry | {m['gone_pct'].median():.1%} | {m['gone_pct'].quantile(.75):.1%} | {m['gone_pct'].quantile(.9):.1%} |" if len(m) else "",
        f"| ATR(15m) at entry, % of price | {m['atr_pct'].median():.2%} | {m['atr_pct'].quantile(.75):.2%} | {m['atr_pct'].quantile(.9):.2%} |" if len(m) else "",
        f"| drawdown after entry, ATR | {pct.get('p50', 0):.2f} | {pct.get('p75', 0):.2f} | {pct.get('p90', 0):.2f} |",
        f"| move from entry to peak, ATR | {m['peak_atr'].median():.2f} | {m['peak_atr'].quantile(.75):.2f} | {m['peak_atr'].quantile(.9):.2f} |" if len(m) else "",
        "", "## Trades on all out-of-fold Alerts by stop multiplier", "",
        "Entry on the first Alert bar, exit when the Alert is gone for 4 bars, trailing 1.5 ATR; no fees or slippage. "
        "`avg_in_pump` / `avg_outside` split trades by whether they overlap a labeled pump.", "",
        train.md_table(sims.set_index("k")), "",
    ]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(x for x in lines if x is not None))
    print(f"\nЗвіт: {REPORT}")


if __name__ == "__main__":
    main()
