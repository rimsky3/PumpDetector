"""
experiments/style_backtest.py — вхід «у стилі власника» на монетах бектесту (2026-10-02).

experiments/my_entries_profile.py: власник входить у монету, що вже сильно «в грі» (+27% за 24 год, обсяг ×6), але
НЕ на хаях — на відкаті −6% від хаю 24 год, коли останні години спокійніші; модель купує сам сплеск на хаях.
Тут — чи дає таке правило перевагу саме по собі, без моделі, і чи модель поверх нього щось додає.

Правило (на закритті кожного 15m-бару Binance, лише минулі дані; одна угода на монету раз на COOLDOWN):
    гаряча       ціна за 24 год ≥ +R24, обсяг за 4 год ≥ VX × медіани 4-годинного обсягу за 7 днів
    відкат       ціна на D_MIN…D_MAX нижче хаю 24 год
Для контрасту «погоня» (як модель): +R24 і обсяг, але ціна ≤ 1.5% від хаю 24 год і +5% за 4 год.

Вихід однаковий для всіх, щоб порівнювати саме входи: стоп −STOP (типова MAE власника −2.1%; беремо 3%), інакше вихід
через HOLD годин; комісія 0.1%. Плюс ціна через 1 / 4 / 12 год (без стопу) і порівняння з випадковими входами на тих
самих монетах. «Модель увімкнена» — у момент входу рівень Alert (кеш experiments/exit_structure.py).

    python experiments/style_backtest.py
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from features import INTERVAL_15M, aggregate, load_raw

FEE, STOP, HOLD = 0.001, 0.03, 4
COOLDOWN = 16                    # барів (4 год): одна угода на один відкат
CACHE = os.path.join("reports", "backtests", "_exit_structure_cache.pkl")
DATA = os.path.join("reports", "backtests", "_style_bars.pkl")


def signals(h: pd.DataFrame) -> pd.DataFrame:
    c, hi, v = h["price"], h["high"], h["volume"]
    v4 = v.rolling(16).sum()
    norm = v4.shift(16).rolling(7 * 96, min_periods=5 * 96).median()
    return pd.DataFrame({"r24": c / c.shift(96) - 1, "r4": c / c.shift(16) - 1, "vx": v4 / norm,
                         "dist": c / hi.rolling(96).max() - 1}, index=h.index)


def trade(h: pd.DataFrame, i: int) -> dict | None:
    """Вхід за закриттям бару i; стоп −STOP (перевірка по лоу), інакше вихід через HOLD год."""
    n = HOLD * 4
    if i + n >= len(h):
        return None
    e = h["price"].iloc[i]
    w = h.iloc[i + 1:i + 1 + n]
    hit = np.flatnonzero(w["low"].to_numpy() <= e * (1 - STOP))
    ret = -STOP if len(hit) else w["price"].iloc[-1] / e - 1
    fwd = {f"через {k}г": h["price"].iloc[i + 4 * k] / e - 1 if i + 4 * k < len(h) else np.nan for k in (1, 4, 12)}
    return {"ret": ret - FEE, "stop": bool(len(hit)), "MFE": w["high"].max() / e - 1, **fwd}


def run(bars: dict, alerts: dict, rule) -> pd.DataFrame:
    rows = []
    for b, h in bars.items():
        s = signals(h)
        ok = rule(s).fillna(False).to_numpy()
        al = alerts.get(b)
        last = -10**9
        for i in np.flatnonzero(ok):
            if i - last < COOLDOWN:
                continue
            t = trade(h, i)
            if t is None:
                continue
            last = i
            a = al.get(h.index[i], np.nan) if al is not None else np.nan
            rows.append({"ticker": b, "ts": h.index[i], **t, "alert": bool(a >= 1) if not pd.isna(a) else None})
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"угод": 0}
    v = df["ret"].sort_values()
    gw, gl = v[v > 0].sum(), -v[v <= 0].sum()
    return {"угод": len(v), "монет": df["ticker"].nunique(), "у плюсі": f"{(v > 0).mean():.0%}", "медіана": f"{v.median():+.2%}",
            "сер.": f"{v.mean():+.2%}", "без топ-3": f"{v.iloc[:-3].sum():+.0%}", "PF": f"{gw / gl:.2f}" if gl else "∞",
            "стоп": f"{df['stop'].mean():.0%}", "через 1г": f"{df['через 1г'].median():+.2%}",
            "через 4г": f"{df['через 4г'].median():+.2%}", "через 12г": f"{df['через 12г'].median():+.2%}"}


def main() -> None:
    argparse.ArgumentParser(description="Вхід у стилі власника (гаряча монета + відкат) на монетах бектесту").parse_args()
    load_dotenv(".env")
    cache = pickle.load(open(CACHE, "rb"))            # {монета: (alert_score, бари)} — з experiments/exit_structure.py
    alerts = {b: t["alert_score"] for b, (t, _) in cache.items()}
    if os.path.exists(DATA):
        bars = pickle.load(open(DATA, "rb"))
    else:
        bars = {}
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            for b in sorted(cache):
                h = aggregate(load_raw(conn, [b], INTERVAL_15M, exchanges=["Binance"]), b, "15min")
                if len(h) > 9 * 96:
                    bars[b] = h[["open", "high", "low", "price", "volume"]]
        pickle.dump(bars, open(DATA, "wb"))
    start = min(t.index.min() for t in alerts.values())          # той самий період, що й бектест моделі
    bars = {b: h[h.index >= start - pd.Timedelta(days=8)] for b, h in bars.items()}
    print(f"монет: {len(bars)}; вихід: стоп −{STOP:.0%} або через {HOLD} год; комісія {FEE:.1%}\n")

    def in_window(df):
        return df[df["ts"] >= start] if len(df) else df
    rules = {
        "стиль: +20%/24г, обсяг ×4, −3…−10% від хаю": lambda s: (s.r24 >= .20) & (s.vx >= 4) & (s.dist <= -.03) & (s.dist >= -.10),
        "стиль: +15%/24г": lambda s: (s.r24 >= .15) & (s.vx >= 4) & (s.dist <= -.03) & (s.dist >= -.10),
        "стиль: +30%/24г": lambda s: (s.r24 >= .30) & (s.vx >= 4) & (s.dist <= -.03) & (s.dist >= -.10),
        "стиль: глибший відкат −5…−15%": lambda s: (s.r24 >= .20) & (s.vx >= 4) & (s.dist <= -.05) & (s.dist >= -.15),
        "погоня: +20%/24г, на хаях, +5%/4г": lambda s: (s.r24 >= .20) & (s.vx >= 4) & (s.dist >= -.015) & (s.r4 >= .05),
        "випадковий вхід (ті самі монети)": None,
    }
    res = {}
    rng = np.random.default_rng(0)
    for name, rule in rules.items():
        if rule is None:
            rr = []
            for b, h in bars.items():
                idx = np.flatnonzero(h.index >= start)
                for i in rng.choice(idx[:-HOLD * 4 - 1], size=min(20, max(0, len(idx) - HOLD * 4 - 1)), replace=False):
                    t = trade(h, i)
                    if t:
                        rr.append({"ticker": b, "ts": h.index[i], **t, "alert": None})
            df = pd.DataFrame(rr)
        else:
            df = in_window(run(bars, alerts, rule))
        res[name] = stats(df)
        if rule is not None and len(df) and df["alert"].notna().any():
            res["    модель: Alert горить"] = stats(df[df["alert"] == True])  # noqa: E712
            res["    модель: Alert не горить"] = stats(df[df["alert"] == False])  # noqa: E712
            res[f"    {name[:5]}·Alert горить"] = res.pop("    модель: Alert горить")
            res[f"    {name[:5]}·Alert не горить"] = res.pop("    модель: Alert не горить")
    pd.set_option("display.width", 260)
    print(pd.DataFrame(res).T.fillna("").to_string())


if __name__ == "__main__":
    main()
