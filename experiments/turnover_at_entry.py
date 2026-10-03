"""
experiments/turnover_at_entry.py — найбільші профіти за Alert проти обороту монети в момент входу (2026-10-03).

Питання власника: монети з оборотом < 20 млн $ він не торгує. Бектест відбирає монети за СЕРЕДНІМ оборотом за 30 днів
(смуги ≥ 20 млн і 5–20 млн), а в терміналі видно оборот за 24 год у момент сигналу — у гарячої монети він буває
в рази більшим за середній. Тут: оборот Binance (USDT-перп) за 24 год до закриття сигнального бару для кожної угоди
бектесту, топ угод за результатом і результат за кошиками обороту.

    python experiments/turnover_at_entry.py     # → reports/turnover_at_entry.md
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from features import INTERVAL_15M, aggregate, load_raw

RUNS = {"робоча": ["20261003_1213_current_market", "20261003_1219_current_market"],
        "кандидат": ["20261003_1216_candidate_market_k2.48", "20261003_1224_candidate_market_k2.48"]}
BUCKETS = [0, 10e6, 20e6, 50e6, 100e6, 300e6, np.inf]
OUT = os.path.join("reports", "turnover_at_entry.md")


def pf(v: pd.Series) -> float:
    v = v.dropna()
    gl = -v[v < 0].sum()
    return round(v[v > 0].sum() / gl, 2) if gl else np.nan


def money(x: float) -> str:
    return f"{x / 1e6:.0f} млн" if x >= 1e6 else f"{x / 1e3:.0f} тис."


def main() -> None:
    argparse.ArgumentParser(description="Профіти за Alert проти обороту монети за 24 год у момент входу (бектести 03.10).").parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    rows = []
    for model, runs in RUNS.items():
        for run in runs:
            d = json.load(open(os.path.join("reports", "backtests", run + ".json"), encoding="utf-8"))
            avg = {u["base"]: u["turnover"] for u in d["meta"]["universe"]}
            band = "≥ 20 млн" if d["meta"]["min_turnover"] >= 20e6 else "5–20 млн"
            for t in d["trades_list"]:
                if t["reason"] == "open" or t["ret"] is None or (isinstance(t["ret"], float) and np.isnan(t["ret"])):
                    continue
                rows.append({"model": model, "band": band, "ticker": t["ticker"], "entry_ts": t["entry_ts"], "exit_ts": t["exit_ts"],
                             "ret": t["ret"], "reason": t["reason"], "avg30": avg.get(t["ticker"], np.nan)})
    tr = pd.DataFrame(rows)
    tr["entry"] = pd.to_datetime(tr["entry_ts"], unit="s", utc=True)
    vol = {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        since = tr["entry"].min() - pd.Timedelta(days=2)
        for n, b in enumerate(sorted(tr["ticker"].unique()), 1):
            h = aggregate(load_raw(conn, [b], INTERVAL_15M, since=since.to_pydatetime(), exchanges=["Binance"]), b, "15min")
            if not h.empty:
                vol[b] = h["volume"].fillna(0).rolling(96, min_periods=90).sum()   # 24 год до закриття бару включно
            if n % 40 == 0:
                print(f"  {n} монет", flush=True)
    tr["turn24"] = [vol[b].get(ts, np.nan) if b in vol else np.nan for b, ts in zip(tr["ticker"], tr["entry"])]
    tr["x_avg"] = tr["turn24"] / tr["avg30"]
    tr["hold_h"] = (tr["exit_ts"] - tr["entry_ts"]) / 3600

    md = ["# Профіти за Alert проти обороту в момент входу", "",
          "Бектести 03.10 (вересень, вхід по ринку, стоп 2.48 ATR, частковий вихід); **оборот 24 год** — Binance USDT-перп за добу до "
          "закриття сигнального бару; **сер. 30 д** — середній денний оборот, за яким бектест відбирав монети.", ""]
    for model in RUNS:
        x = tr[tr["model"] == model]
        top = x.sort_values("ret", ascending=False).head(25)
        md += [f"## {model}: топ-25 угод", "", "| тікер | вхід (UTC) | результат | оборот 24 год | сер. 30 д | × до середнього | смуга (30 д) | утримання, год |",
               "|---|---|---|---|---|---|---|---|"]
        for r in top.itertuples():
            md.append(f"| {r.ticker} | {r.entry:%m-%d %H:%M} | {r.ret:+.1%} | {money(r.turn24) if r.turn24 == r.turn24 else '—'} | "
                      f"{money(r.avg30)} | {r.x_avg:.1f}× | {r.band} | {r.hold_h:.1f} |")
        md += [""]
        x = x.assign(bucket=pd.cut(x["turn24"], BUCKETS, labels=["< 10 млн", "10–20", "20–50", "50–100", "100–300", "> 300"], right=False))
        rows_ = []
        for bkt, g in x.groupby("bucket", observed=True):
            v = g["ret"]
            rows_.append(f"| {bkt} | {len(g)} | {g['ticker'].nunique()} | {(v > 0).mean():.0%} | {pf(v)} | {pf(v.sort_values().iloc[:-3]) if len(v) > 3 else '—'} | "
                         f"{v.median():+.2%} | {(v >= 0.10).sum()} | {v[v >= 0.10].sum():+.0%} |")
        big = x[x["ret"] >= 0.10]
        md += [f"## {model}: за оборотом 24 год у момент входу", "",
               "| оборот 24 год | угод | монет | у плюсі | PF | без топ-3 | медіана | угод ≥ +10% | їх сума |", "|---|---|---|---|---|---|---|---|---|"] + rows_
        for lbl, m in (("оборот 24 год ≥ 20 млн (будь-яка смуга)", x["turn24"] >= 20e6), ("смуга ≥ 20 млн (сер. 30 д)", x["band"] == "≥ 20 млн"),
                       ("обидва: сер. 30 д ≥ 20 і 24 год ≥ 20", (x["band"] == "≥ 20 млн") & (x["turn24"] >= 20e6))):
            v = x.loc[m, "ret"]
            md.append(f"| **{lbl}** | {len(v)} | {x.loc[m, 'ticker'].nunique()} | {(v > 0).mean():.0%} | {pf(v)} | {pf(v.sort_values().iloc[:-3])} | "
                      f"{v.median():+.2%} | {(v >= 0.10).sum()} | {v[v >= 0.10].sum():+.0%} |")
        md += ["", f"Угоди ≥ +10% ({len(big)}): оборот 24 год — медіана {money(big['turn24'].median())}, у {big['x_avg'].median():.1f}× "
               f"більший за середній за 30 д; з них {(big['turn24'] >= 20e6).mean():.0%} мали ≥ 20 млн за 24 год, "
               f"{(big['band'] == '≥ 20 млн').mean():.0%} — зі смуги ≥ 20 млн за середнім.", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(md))
    tr.to_csv(os.path.join("reports", "turnover_at_entry.csv"), index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
