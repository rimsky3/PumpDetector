"""
experiments/backtest_slices.py — розріз прогону backtest.py: (1) з фільтром сканера на вході, (2) за оборотом монети.

(1) Наживо модель бачить лише тікери, які сканер додав у список («прокидання»). Угода рахується, лише якщо в момент
    входу сканер із порогами набору тримав би монету в списку: було спрацювання фільтра за останні ttl_hours
    (scanner.passes_series на Binance 15m з бази; без продовження сигналом моделі — консервативно).
(2) Смуги середнього денного обороту монети (з універсуму прогону).

Метрики як у вкладці «Бектест»: угоди, % у плюсі, середня, медіана, сума, сума без топ-3, угод на день.

    python experiments/backtest_slices.py                      # останній прогін
    python experiments/backtest_slices.py --run 20261001_1441_current_alert
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import scanner
from features import INTERVAL_15M, aggregate, load_raw


def stats(t: pd.DataFrame, days: int) -> dict:
    c = t[t["reason"] != "open"]
    r = c["ret"].sort_values()
    if r.empty:
        return {"угод": 0}
    return {"угод": len(r), "монет": c["ticker"].nunique(), "у плюсі": f"{(r > 0).mean():.0%}", "сер.": f"{r.mean():+.2%}",
            "медіана": f"{r.median():+.2%}", "сума": f"{r.sum():+.0%}", "без топ-3": f"{r.iloc[:-3].sum():+.0%}" if len(r) > 3 else "—",
            "угод/день": f"{len(r) / days:.1f}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Розріз бектесту: фільтр сканера і смуги обороту (експеримент)")
    ap.add_argument("--run", default=None, help="id прогону (за замовчуванням останній у reports/backtests)")
    args = ap.parse_args()
    load_dotenv(".env")
    files = sorted(glob.glob(os.path.join("reports", "backtests", "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith("_")]
    path = os.path.join("reports", "backtests", args.run + ".json") if args.run else files[-1]
    d = json.load(open(path, encoding="utf-8"))
    m, days = d["meta"], d["meta"]["days"]
    t = pd.DataFrame(d["trades_list"])
    t["entry"] = pd.to_datetime(t["entry_ts"], unit="s", utc=True)
    turn = {u["base"]: u["turnover"] for u in m["universe"]}
    t["turnover"] = t["ticker"].map(turn)
    print(f"прогін {m['id']}: {d['coins']} монет, {len(t)} угод\n")

    rows = {"усі угоди (як у вкладці)": stats(t, days)}
    # (1) фільтр сканера
    cfg = scanner.load_config()
    since = pd.Timestamp(m["at"]) - pd.Timedelta(days=days + 8)
    sets = {"Широко (5 млн)": scanner.PRESETS["wide"], "Баланс (20 млн)": scanner.PRESETS["balance"], "Суворо (30 млн)": scanner.PRESETS["strict"]}
    in_list = {k: np.zeros(len(t), dtype=bool) for k in sets}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for tk, idx in t.groupby("ticker").groups.items():
            h = aggregate(load_raw(conn, [tk], INTERVAL_15M, since=since, exchanges=["Binance"]), tk, "15min")
            if h.empty:
                continue
            for k, p in sets.items():
                ok = scanner.passes_series(h, {**p, "max_oi": cfg["max_oi"]}, 4).fillna(False)
                trig = (ok.index + pd.Timedelta(minutes=15))[ok.to_numpy()]
                if len(trig) == 0:
                    continue
                trig = np.sort(trig.tz_convert(None).to_numpy())
                for i in idx:
                    e = t.at[i, "entry"].tz_convert(None).to_datetime64()
                    j = np.searchsorted(trig, e, side="right") - 1          # останнє спрацювання не пізніше входу
                    in_list[k][t.index.get_loc(i)] = j >= 0 and (e - trig[j]) <= np.timedelta64(int(cfg["ttl_hours"] * 3600), "s")
    for k in sets:
        rows[f"фільтр сканера: {k}"] = stats(t[in_list[k]], days)
        rows[f"  поза списком сканера ({k})"] = stats(t[~in_list[k]], days)
    # (2) смуги обороту
    for lo, hi, name in ((0, 50e6, "оборот 20–50 млн $"), (50e6, 150e6, "оборот 50–150 млн $"), (150e6, 1e13, "оборот 150+ млн $")):
        rows[name] = stats(t[(t["turnover"] >= lo) & (t["turnover"] < hi)], days)
    print(pd.DataFrame(rows).T.fillna("").to_string())


if __name__ == "__main__":
    main()
