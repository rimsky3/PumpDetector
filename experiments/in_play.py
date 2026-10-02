"""
experiments/in_play.py — чи працює модель на монетах «в грі» і не працює на «мертвих» (гіпотеза власника, 2026-10-01).

Угоди беруться з прогонів backtest.py (монети поза навчанням). Для кожної угоди — чи була монета «в грі» на момент входу,
лише за даними ДО входу (Binance 15m з бази → години). Ознаки «в грі» (так їх бачить власник), по ковзному вікну 24 год:

    обсяг ×N     оборот за 24 год / медіана такого обороту за попередні 30 днів ≥ vol_x
    OI днями     OI зріс за 3 дні ≥ oi_3d
    волатильність діапазон (high − low) / ціна за 24 год / медіана за 30 днів ≥ range_x
    топ руху     |зміна ціни за 24 год| — у верхніх top_pct серед усіх монет обох прогонів у ту саму годину

Монета «в грі», якщо будь-яка ознака спрацювала за останні STAY_DAYS днів (власник: 3–5 днів). Два варіанти:
    з поточним рухом     вікно до входу включно (спрацювання може дати сам рух, на який входимо)
    до поточного руху    вікно закінчується за EXCLUDE_H год до входу — монета була в грі ще ДО цього сплеску

Порівняння груп: угоди, монети, % у плюсі, середня, медіана, сума без топ-3. Пороги — кілька наборів (стійкість, а не
підбір найкращого).

    python experiments/in_play.py
    python experiments/in_play.py --runs 20261001_1441_current_alert 20261001_1517_current_alert
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

STAY_DAYS = 5
EXCLUDE_H = 6
SETS = {"м'який": dict(vol_x=2.0, oi_3d=0.15, range_x=1.5, top_pct=0.15),
        "середній": dict(vol_x=3.0, oi_3d=0.25, range_x=2.0, top_pct=0.10),
        "жорсткий": dict(vol_x=5.0, oi_3d=0.40, range_x=3.0, top_pct=0.05)}


def hourly(h: pd.DataFrame) -> pd.DataFrame:
    """15m-бари Binance → години (індекс — ЗАКРИТТЯ години: усе, що відомо на цей момент)."""
    g = h.resample("h").agg({"volume": "sum", "price": "last", "high": "max", "low": "min", "oi": "last"})
    g.index = g.index + pd.Timedelta(hours=1)
    return g.dropna(subset=["price"])


def signals(g: pd.DataFrame) -> pd.DataFrame:
    """Ковзні 24-годинні метрики на кожну годину (лише минуле)."""
    vol24 = g["volume"].rolling(24, min_periods=20).sum()
    rng24 = (g["high"].rolling(24).max() - g["low"].rolling(24).min()) / g["price"]
    # норма — медіана за 30 днів ДО останньої доби (сплеск не рахує сам себе); мінімум 7 днів історії
    norm_v = vol24.shift(24).rolling(30 * 24, min_periods=7 * 24).median()
    norm_r = rng24.shift(24).rolling(30 * 24, min_periods=7 * 24).median()
    return pd.DataFrame({"vol_x": vol24 / norm_v, "range_x": rng24 / norm_r,
                         "oi_3d": g["oi"] / g["oi"].shift(72) - 1, "ret24": g["price"] / g["price"].shift(24) - 1})


def stats(t: pd.DataFrame) -> dict:
    r = t["ret"].sort_values()
    if r.empty:
        return {"угод": 0}
    return {"угод": len(r), "монет": t["ticker"].nunique(), "у плюсі": f"{(r > 0).mean():.0%}", "сер.": f"{r.mean():+.2%}",
            "медіана": f"{r.median():+.2%}", "сума": f"{r.sum():+.0%}", "без топ-3": f"{r.iloc[:-3].sum():+.0%}" if len(r) > 3 else "—"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Угоди бектесту: монета «в грі» проти «мертвої» (експеримент)")
    ap.add_argument("--runs", nargs="+", default=["20261001_1441_current_alert", "20261001_1517_current_alert"],
                    help="id прогонів у reports/backtests")
    args = ap.parse_args()
    load_dotenv(".env")
    trades, coins = [], set()
    for run in args.runs:
        d = json.load(open(os.path.join("reports", "backtests", run + ".json"), encoding="utf-8"))
        t = pd.DataFrame(d["trades_list"]).assign(run=run)
        trades.append(t[t["reason"] != "open"])
        coins |= {u["base"] for u in d["meta"]["universe"]}
    t = pd.concat(trades, ignore_index=True)
    t["entry"] = pd.to_datetime(t["entry_ts"], unit="s", utc=True)
    coins |= set(t["ticker"])

    met = {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for tk in sorted(coins):
            h = aggregate(load_raw(conn, [tk], INTERVAL_15M, exchanges=["Binance"]), tk, "15min")
            if len(h) > 7 * 96:
                met[tk] = signals(hourly(h))
    # топ руху: перцентиль |зміни за 24 год| серед усіх монет у ту саму годину
    ret = pd.DataFrame({tk: m["ret24"].abs() for tk, m in met.items()})
    rank = ret.rank(axis=1, pct=True)
    print(f"угод: {len(t)} на {t['ticker'].nunique()} монетах; метрики для {len(met)} монет "
          f"(ранжування в середньому серед {ret.notna().sum(axis=1).median():.0f} монет за годину)\n")

    rows = {"усі угоди": stats(t)}
    for name, p in SETS.items():
        for mode, lag in (("з поточним рухом", 0), ("до поточного руху", EXCLUDE_H)):
            flags = {k: np.zeros(len(t), dtype=bool) for k in ("any", "vol", "oi", "range", "top")}
            for tk, idx in t.groupby("ticker").groups.items():
                if tk not in met:
                    continue
                m = met[tk]
                fired = pd.DataFrame({"vol": m["vol_x"] >= p["vol_x"], "oi": m["oi_3d"] >= p["oi_3d"],
                                      "range": m["range_x"] >= p["range_x"],
                                      "top": rank[tk].reindex(m.index) >= 1 - p["top_pct"]})
                # «в грі» на момент T: спрацювання в будь-яку годину з [T − STAY_DAYS, T − lag]
                win = fired.astype(float).rolling(f"{STAY_DAYS * 24 - lag}h").max().shift(lag).fillna(0).astype(bool)
                for i in idx:
                    at = win.index.searchsorted(t.at[i, "entry"], side="right") - 1      # остання закрита година
                    if at < 0:
                        continue
                    row = win.iloc[at]
                    k = t.index.get_loc(i)
                    for c in ("vol", "oi", "range", "top"):
                        flags[c][k] = row[c]
                    flags["any"][k] = row.any()
            rows[f"{name} · {mode}: В ГРІ"] = stats(t[flags["any"]])
            rows[f"{name} · {mode}: мертва"] = stats(t[~flags["any"]])
            if name == "середній":
                for c, lab in (("vol", "обсяг ×N"), ("oi", "OI днями"), ("range", "волатильність"), ("top", "топ руху")):
                    rows[f"  {mode}: лише ознака «{lab}»"] = stats(t[flags[c]])
                    rows[f"  {mode}: без ознаки «{lab}»"] = stats(t[~flags[c]])
    print(pd.DataFrame(rows).T.fillna("").to_string())

    # де губиться перевага: вхід чи вихід. MFE/MAE за 4 і 12 год після входу (у % від ціни входу) — угоди моделі
    # проти ВИПАДКОВИХ входів на тих самих монетах у той самий період (по 5 на кожну угоду, seed 0)
    rng = np.random.default_rng(0)
    bars = {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for tk in t["ticker"].unique():
            bars[tk] = aggregate(load_raw(conn, [tk], INTERVAL_15M, exchanges=["Binance"]), tk, "15min")[["price", "high", "low"]]

    def excursions(tk, ts):
        h = bars[tk]
        i = h.index.searchsorted(ts)        # бар входу (індекс — відкриття, ціна — закриття, як у бектесті)
        if i >= len(h) - 48:
            return None
        p0 = h["price"].iloc[i]
        out = {}
        for n, lab in ((16, "4г"), (48, "12г")):
            w = h.iloc[i + 1:i + 1 + n]
            out[f"MFE {lab}"], out[f"MAE {lab}"] = w["high"].max() / p0 - 1, w["low"].min() / p0 - 1
            out[f"ціна через {lab}"] = w["price"].iloc[-1] / p0 - 1
        return out
    model_ex = [e for e in (excursions(r.ticker, r.entry) for r in t.itertuples()) if e]
    rand_ex = []
    for r in t.itertuples():
        h = bars[r.ticker]
        lo, hi = t["entry"].min(), h.index.max() - pd.Timedelta(hours=13)
        cand = h.index[(h.index >= lo) & (h.index <= hi)]
        for ts in rng.choice(cand, 5) if len(cand) else []:
            e = excursions(r.ticker, pd.Timestamp(ts))
            if e:
                rand_ex.append(e)
    cmp = pd.DataFrame({"входи моделі": pd.DataFrame(model_ex).median(), "випадкові входи": pd.DataFrame(rand_ex).median()})
    print(f"\nМедіани після входу (моделі: {len(model_ex)}, випадкових: {len(rand_ex)}):")
    print(cmp.map(lambda v: f"{v:+.2%}").to_string())
    print(f"\n«В грі» — будь-яка ознака за {STAY_DAYS} днів до входу; «до поточного руху» — без останніх {EXCLUDE_H} год.")
    for name, p in SETS.items():
        print(f"  {name}: обсяг ×{p['vol_x']}, OI 3 дні +{p['oi_3d']:.0%}, діапазон ×{p['range_x']}, топ {p['top_pct']:.0%} руху")


if __name__ == "__main__":
    main()
