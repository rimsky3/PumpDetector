"""
experiments/in_position.py — що всередині відкритої позиції моделі говорить про силу імпульсу.

Позиції — як у роботі: вхід на першому Alert (out-of-fold, models/oof_15m.parquet), вихід — Alert згас 4 бари або стоп.
Для кожного 15m-бару всередині позиції — що робить ціна далі (1 / 2 / 4 год: дохідність від закриття бару,
найгірше просідання, чи буде новий максимум руху за 4 год). Порівнюємо типи барів:

    A. «повторне спалахування»: Alert знову загорівся після 1–3 барів без нього (позиція відкрита) — точка добору?
       проти «Alert горить далі» (бар з Alert після бару з Alert) і проти самого першого входу.
    B. «імпульс слабшає»: 15m-оцінка падає 3 бари поспіль і ціна 4 бари не оновлює максимум руху — сигнал виходу?
       проти решти барів позиції. Плюс угоди: вихід як зараз проти «як зараз АБО слабшає».

Чесність: медіани, а для угод — ще й сума без 2 найбільших (урок exit_examples.py).

    python experiments/in_position.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from features import INTERVAL_15M, aggregate, load_raw
from risk import atr

FEE = 0.001


def walk(h, score, s15, k, fade_exit=False, exit_bars=4):
    """Позиції + позначки барів усередині. Повертає (угоди, бари)."""
    a = atr(h).to_numpy()
    sc, s = score.reindex(h.index).to_numpy(), s15.reindex(h.index).to_numpy()
    px, hi, lo = h["price"].to_numpy(), h["high"].to_numpy(), h["low"].to_numpy()
    n = len(px)
    trades, bars, pos, prev_on = [], [], None, False
    for i in range(n):
        on = bool(sc[i] >= 1) if not np.isnan(sc[i]) else False
        if pos is None:
            if on and not prev_on and not np.isnan(a[i]):
                pos = {"i0": i, "e": px[i], "stop": px[i] - k * a[i], "pk": hi[i], "below": 0, "pk_i": i}
                bars.append({"i": i, "kind": "first"})
            prev_on = on
            continue
        if lo[i] <= pos["stop"]:
            trades.append((pos, pos["stop"], i, "stop")); pos = None; prev_on = on; continue
        if hi[i] > pos["pk"]:
            pos["pk"], pos["pk_i"] = hi[i], i
        was_below = pos["below"]
        pos["below"] = 0 if on else pos["below"] + 1
        fading = (i >= 3 and not np.isnan(s[i - 3:i + 1]).any() and s[i] < s[i - 1] < s[i - 2] < s[i - 3]
                  and i - pos["pk_i"] >= 4)
        kind = "reignite" if on and 1 <= was_below <= 3 else "continue" if on and was_below == 0 else None
        if kind:
            bars.append({"i": i, "kind": kind})
        bars.append({"i": i, "kind": "fading" if fading else "other_in_pos"})
        if pos["below"] >= exit_bars or (fade_exit and fading):
            trades.append((pos, px[i], i, "fade" if fading and pos["below"] < exit_bars else "rule")); pos = None
        prev_on = on
    # наслідки для барів
    out = []
    for b in bars:
        i = b["i"]
        if i + 16 >= n:
            continue
        c = px[i]
        run_pk = max(hi[max(0, i - 200):i + 1])
        out.append({"kind": b["kind"], "r1": px[i + 4] / c - 1, "r2": px[i + 8] / c - 1, "r4": px[i + 16] / c - 1,
                    "dd4": lo[i + 1:i + 17].min() / c - 1, "new_high": hi[i + 1:i + 17].max() > run_pk})
    tr = [{"ret": x / p["e"] - 1 - FEE, "why": why, "hold_h": (j - p["i0"]) / 4} for p, x, j, why in trades]
    return tr, out


def main() -> None:
    argparse.ArgumentParser(description="Сигнали всередині відкритої позиції: повторний Alert і «імпульс слабшає»").parse_args()
    load_dotenv(".env")
    oof = pd.read_parquet(os.path.join(train.MODELS_DIR, "oof_15m.parquet"))
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    b15 = joblib.load(train.model_path(train.M15_NAME))
    k, share = b15["risk"]["k"], b15.get("confirm", {}).get("share", 0.75)
    oof["s15"] = oof["p15"] / oof["thr15"]
    oof["score"] = np.minimum(oof["s15"], oof["s1h"].fillna(0) / share)
    bars, tr_cur, tr_fade = [], [], []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for b in sorted(oof["base_asset"].unique()):
            h = aggregate(load_raw(conn, [b], INTERVAL_15M), b, "15min").dropna(subset=["price", "high", "low"])
            g = oof[oof["base_asset"] == b].set_index("ts")
            t, bb = walk(h, g["score"], g["s15"], k)
            tr_cur += t; bars += bb
            tr_fade += walk(h, g["score"], g["s15"], k, fade_exit=True)[0]
    d = pd.DataFrame(bars)
    agg = d.groupby("kind").agg(барів=("r1", "size"), r1=("r1", "median"), r2=("r2", "median"), r4=("r4", "median"),
                                r4_mean=("r4", "mean"), r4_plus=("r4", lambda x: (x > 0).mean()), dd4=("dd4", "median"),
                                new_high=("new_high", "mean"))
    agg = agg.reindex(["first", "reignite", "continue", "fading", "other_in_pos"])
    fmt = agg.copy()
    for c in ("r1", "r2", "r4", "r4_mean", "dd4"):
        fmt[c] = agg[c].map(lambda v: f"{v:+.2%}")
    for c in ("r4_plus", "new_high"):
        fmt[c] = agg[c].map(lambda v: f"{v:.0%}")
    fmt.columns = ["барів", "через 1 год (мед.)", "2 год", "4 год", "4 год (сер.)", "4 год > 0", "просідання 4 год", "новий максимум ≤ 4 год"]
    print("Що робить ціна після бару всередині позиції:\n" + fmt.to_string())

    print("\nУгоди: вихід як зараз проти «як зараз АБО імпульс слабшає»:")
    for name, tr in (("як зараз", tr_cur), ("+ слабшає", tr_fade)):
        r = pd.Series([x["ret"] for x in tr]).sort_values()
        print(f"  {name:<10} угод {len(r)} · у плюсі {(r > 0).mean():.0%} · медіана {r.median():+.2%} · сер. {r.mean():+.2%}"
              f" · сума {r.sum():+.0%} · без 2 найбільших {r.iloc[:-2].sum():+.0%} · утримання {np.median([x['hold_h'] for x in tr]):.1f} год")


if __name__ == "__main__":
    main()
