"""
experiments/my_entries_profile.py — чим вхід власника відрізняється від входу моделі (2026-10-02).

Угоди власника — my_trades.py (Binance Futures, лише читання); «системні» = тримав ≥ 10 хв і без ознак тільту
(«відігратися» ≤ 5 хв після збитку, «серія» 5+ входів за 20 хв) — відбір НЕ за результатом. Беруться лонги.
Входи моделі — прогони backtest.py (Alert, монети поза навчанням).

Метрики — лише ціна й обсяг Binance 15m (історія свічок є за будь-яку дату; OI Binance — лише 30 днів), усе ДО входу:
    ret 1h / 4h / 24h       зміна від ціни 1, 4, 24 год тому до ціни входу
    до хаю 24h / 7d         ціна входу відносно максимуму за 24 год / 7 днів («скільки місця над головою»)
    пробій 7d               вхід вище максимуму попередніх 7 днів (без останньої години)
    відкат                  де вхід між лоу 24 год (1) і хаєм 12 год (0): 0 — на хаях, 1 — на дні
    обсяг ×                 обсяг за 4 год / медіана 4-годинного обсягу за 7 днів до того
Під час угоди: MFE / MAE (найкраща / найгірша ціна до виходу), «взято руху» — (вихід − вхід) / (максимум − вхід).

    python experiments/my_entries_profile.py
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from exchanges import binance_klines
from features import INTERVAL_15M, aggregate, load_raw

B15, H = pd.Timedelta(minutes=15), pd.Timedelta(hours=1)
CACHE = os.path.join("reports", "my_trades", "_klines15.pkl")


def metrics(h: pd.DataFrame, T: pd.Timestamp, E: float, X: float | None, TX: pd.Timestamp | None) -> dict | None:
    """h — 15m-бари (індекс — відкриття; o h l c v). T — момент входу, E — ціна входу; X, TX — вихід."""
    close_t = h.index + B15
    pre = h[close_t <= T]
    if len(pre) < 7 * 96 or pre.index[-1] + B15 < T - H:
        return None

    def c_at(t):
        w = pre[pre.index + B15 <= t]
        return w["c"].iloc[-1] if len(w) else np.nan

    def last(dt):
        return pre[pre.index + B15 > T - dt]
    hi24, lo24, hi12 = last(24 * H)["h"].max(), last(24 * H)["l"].min(), last(12 * H)["h"].max()
    prior7 = pre[(pre.index + B15 > T - 7 * 24 * H) & (pre.index + B15 <= T - H)]
    v4 = pre["v"].rolling(16).sum()
    norm = v4.iloc[:-16].tail(7 * 96).median()
    out = {"ret 1h": E / c_at(T - H) - 1, "ret 4h": E / c_at(T - 4 * H) - 1, "ret 24h": E / c_at(T - 24 * H) - 1,
           "до хаю 24h": E / hi24 - 1, "до хаю 7d": E / last(7 * 24 * H)["h"].max() - 1,
           "пробій 7d": float(E > prior7["h"].max()) if len(prior7) else np.nan,
           "відкат": (hi12 - E) / (hi12 - lo24) if hi12 > lo24 else np.nan,
           "обсяг ×": v4.iloc[-1] / norm if norm and norm > 0 else np.nan}
    if X is not None and TX is not None:
        dur = h[(h.index + B15 > T) & (h.index < TX)]
        if len(dur):
            mx, mn = max(dur["h"].max(), X), min(dur["l"].min(), X)
            out.update({"MFE": mx / E - 1, "MAE": mn / E - 1, "взято руху": (X - E) / (mx - E) if mx > E else np.nan})
    return out


def main() -> None:
    argparse.ArgumentParser(description="Профіль входів власника проти входів моделі (експеримент)").parse_args()
    load_dotenv(".env")
    p = pd.read_parquet(os.path.join("reports", "my_trades", "positions.parquet"))
    p = p[p["exit_time"].notna() & ~(p["revenge"] | p["burst"]) & (p["minutes"] >= 10) & (p["side"] == "long")]
    # 15m-свічки Binance навколо кожної угоди (кеш: повторний запуск не ходить у біржу)
    cache = pd.read_pickle(CACHE) if os.path.exists(CACHE) else {}
    for r in p.itertuples():
        key = (r.symbol, r.entry_time.floor("D"))
        if key in cache:
            continue
        start = int((r.entry_time - pd.Timedelta(days=8)).timestamp() * 1000)
        end = int((max(r.exit_time, r.entry_time) + pd.Timedelta(hours=6)).timestamp() * 1000)
        try:
            k = binance_klines(r.symbol, start, end, "15m").set_index("ts")
            cache[key] = k.rename(columns={})[["o", "h", "l", "c", "v"]]
        except Exception as e:
            print(f"  ! {r.symbol}: {e}")
            cache[key] = pd.DataFrame()
        time.sleep(0.1)
    pd.to_pickle(cache, CACHE)
    mine = []
    for r in p.itertuples():
        h = cache.get((r.symbol, r.entry_time.floor("D")))
        if h is None or h.empty:
            continue
        m = metrics(h, r.entry_time, r.entry_px, r.exit_px, r.exit_time)
        if m:
            mine.append({**m, "win": r.net > 0, "ret": r.ret})
    mine = pd.DataFrame(mine)

    # входи моделі з бектесту (15m Binance з бази)
    model = []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for path in sorted(glob.glob(os.path.join("reports", "backtests", "2026*_alert.json"))):
            d = json.load(open(path, encoding="utf-8"))
            t = pd.DataFrame(d["trades_list"])
            t = t[t["reason"] != "open"]
            for tk, g in t.groupby("ticker"):
                a = aggregate(load_raw(conn, [tk], INTERVAL_15M, exchanges=["Binance"]), tk, "15min")
                if a.empty:
                    continue
                h = a.rename(columns={"open": "o", "high": "h", "low": "l", "price": "c", "volume": "v"})[["o", "h", "l", "c", "v"]]
                for r in g.itertuples():
                    T = pd.Timestamp(r.entry_ts, unit="s", tz="UTC") + B15
                    TX = pd.Timestamp(r.exit_ts, unit="s", tz="UTC") + B15
                    m = metrics(h, T, r.entry, r.exit, TX)
                    if m:
                        model.append({**m, "win": r.ret > 0, "ret": r.ret})
    model = pd.DataFrame(model).drop_duplicates()

    def col(df):
        q = df.drop(columns=["win", "ret"]).median()
        q["пробій 7d"] = df["пробій 7d"].mean()
        return q
    table = pd.DataFrame({
        f"твої, усі ({len(mine)})": col(mine), f"твої в плюсі ({mine['win'].sum()})": col(mine[mine["win"]]),
        f"твої в мінусі ({(~mine['win']).sum()})": col(mine[~mine["win"]]),
        f"модель, усі ({len(model)})": col(model), f"модель в плюсі ({model['win'].sum()})": col(model[model["win"]]),
        f"модель в мінусі ({(~model['win']).sum()})": col(model[~model["win"]]),
    })
    fmt = lambda name, v: "—" if pd.isna(v) else (f"{v:.0%}" if name in ("пробій 7d", "взято руху") else  # noqa: E731
                                                   f"{v:.2f}" if name in ("відкат", "обсяг ×") else f"{v:+.1%}")
    out = table.apply(lambda c: [fmt(i, v) for i, v in c.items()])
    out.index = table.index
    pd.set_option("display.width", 250)
    print("Медіани (пробій 7d — частка угод):\n")
    print(out.to_string())


if __name__ == "__main__":
    main()
