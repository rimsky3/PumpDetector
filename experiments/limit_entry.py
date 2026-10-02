"""
experiments/limit_entry.py — лімітний вхід на відкаті після Alert замість купівлі по ринку (2026-10-02).

Модель купує на закритті сплеску, на хаях (experiments/my_entries_profile.py: −1% від хаю доби; власник — −6%, на відкаті).
Ідея зі статті Kakade, Kearns, Mansour, Ortiz (2004): не бити по ринку, а ставити лімітну заявку біля VWAP.
Правило: на першому барі з Alert (позиції немає) — лімітна заявка на купівлю за ціною L, дійсна K барів:
    ринок            як зараз: вхід за закриттям сигнального бару
    VWAP 1h / 2h     L = min(закриття, VWAP останніх 4 / 8 барів 15m)
    −0.5 / −1 ATR    L = закриття − x·ATR(15m)
Виконання: якщо лоу наступного бару ≤ L — купили за L (або за відкриттям, якщо бар відкрився нижче L). Не виконалась
за K барів — угоди немає (наступна заявка — на наступному епізоді Alert). Після входу — поточне правило позиції
(risk.PARTIAL_EXIT: ½ на +3 ATR, далі беззбиток; решта — 4 бари без Alert або стоп k·ATR від ціни входу). Комісія 0.1%.

Дані — кеші попередніх експериментів (без бази): оцінки Alert робочої моделі (experiments/exit_structure.py) і 15m-бари
Binance з обсягом (experiments/style_backtest.py), 219 монет поза навчанням, 30 днів.

    python experiments/limit_entry.py
"""

import argparse
import itertools
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

import train
from risk import PARTIAL_EXIT, atr

FEE = 0.001
ALERTS = os.path.join("reports", "backtests", "_exit_structure_cache.pkl")
BARS = os.path.join("reports", "backtests", "_style_bars.pkl")
RUNS = ["20261001_1441_current_alert", "20261001_1517_current_alert"]


def simulate(h: pd.DataFrame, score: pd.Series, k: float, exit_bars: int, mode: str, w: int, x: float, valid: int,
             allow=None) -> list[dict]:
    """allow(i) → bool — додатковий фільтр сигналу на барі i (наприклад, стакан); None — без фільтра."""
    o, hi, lo, c, v = (h[n].to_numpy() for n in ("open", "high", "low", "price", "volume"))
    a = atr(h).to_numpy()
    sc = score.reindex(h.index).to_numpy()
    typical = (hi + lo + c) / 3
    pv, vv = pd.Series(typical * v).rolling(w).sum().to_numpy(), pd.Series(v).rolling(w).sum().to_numpy()
    frac, tp_atr = PARTIAL_EXIT["frac"], PARTIAL_EXIT["atr"]
    out, tr, order, below, prev_alert = [], None, None, 0, False
    for i in range(len(h)):
        p = sc[i]
        alert = not np.isnan(p) and p >= 1
        # 1) заявка чекає виконання
        if tr is None and order is not None:
            if i > order["until"]:
                order = None
            elif lo[i] <= order["px"]:
                fill = min(o[i], order["px"]) if not np.isnan(o[i]) else order["px"]
                tr = {"e": fill, "stop": fill - k * order["atr"], "tp": fill + tp_atr * order["atr"], "part": 0.0, "i0": i,
                      "sig": order["sig"], "wait": i - order["sig"]}
                order, below = None, 0
                # бар виконання: стоп/ціль перевіряються з наступного бару (усередині бару порядок невідомий — консервативно)
                prev_alert = alert
                continue
        # 2) новий сигнал → заявка (або вхід по ринку)
        if tr is None and order is None and alert and not prev_alert and not np.isnan(c[i]) and not np.isnan(a[i]) \
                and (allow is None or allow(i)):
            if mode == "market":
                tr = {"e": c[i], "stop": c[i] - k * a[i], "tp": c[i] + tp_atr * a[i], "part": 0.0, "i0": i, "sig": i, "wait": 0}
                below = 0
                prev_alert = alert
                continue
            px = min(c[i], pv[i] / vv[i]) if mode == "vwap" and vv[i] else c[i] - x * a[i]
            order = {"px": px, "until": i + valid, "atr": a[i], "sig": i}
        prev_alert = alert
        # 3) відкрита позиція
        if tr is not None:
            e = tr["e"]
            if lo[i] <= tr["stop"]:
                rest = tr["stop"] / e - 1
                out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - FEE, "reason": "stop",
                            "sig": tr["sig"], "wait": tr["wait"], "e": e})
                tr = None
                continue
            if not tr["part"] and hi[i] >= tr["tp"]:
                tr["part"] = frac * (tr["tp"] / e - 1)
                tr["stop"] = max(tr["stop"], e)
            below = below + 1 if not alert else 0
            if below >= exit_bars:
                rest = c[i] / e - 1
                out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - FEE, "reason": "model",
                            "sig": tr["sig"], "wait": tr["wait"], "e": e})
                tr = None
    return out


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return round(v[v > 0].sum() / gl, 2) if gl else np.nan


def main() -> None:
    argparse.ArgumentParser(description="Лімітний вхід на відкаті (VWAP / ATR) після Alert замість купівлі по ринку").parse_args()
    m15 = joblib.load(train.model_path(train.M15_NAME))
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    alerts = pickle.load(open(ALERTS, "rb"))
    bars = pickle.load(open(BARS, "rb"))
    band = {}
    for run in RUNS:
        d = json.load(open(os.path.join("reports", "backtests", run + ".json"), encoding="utf-8"))
        for u in d["meta"]["universe"]:
            band[u["base"]] = "≥ 20 млн" if d["meta"]["min_turnover"] >= 20e6 else "5–20 млн"
    coins = sorted(set(alerts) & set(bars))
    variants = {"ринок (як зараз)": ("market", 0, 0, 0)}
    for (name, mode, w, x), valid in itertools.product(
            [("VWAP 1h", "vwap", 4, 0), ("VWAP 2h", "vwap", 8, 0), ("−0.5 ATR", "atr", 0, 0.5), ("−1 ATR", "atr", 0, 1.0)], (4, 8)):
        variants[f"{name}, діє {valid // 4} год"] = (mode, max(w, 1), x, valid)
    res, signals = {}, None
    for name, (mode, w, x, valid) in variants.items():
        rows = []
        for b in coins:
            t, _ = alerts[b]
            h = bars[b]
            h = h[h.index >= t.index.min() - pd.Timedelta(days=1)]
            for r in simulate(h, t["alert_score"], k, exit_bars, mode, w, x, valid):
                rows.append({**r, "ticker": b, "band": band.get(b, "")})
        df = pd.DataFrame(rows)
        if name.startswith("ринок"):
            signals = df.set_index(["ticker", "sig"])["ret"]
        v = df["ret"].sort_values()
        got = df.set_index(["ticker", "sig"])["ret"]
        big = signals[signals >= 0.10]
        res[name] = {"угод": len(v), "виконано": f"{len(v) / len(signals):.0%}", "у плюсі": f"{(v > 0).mean():.0%}",
                     "медіана": f"{v.median():+.2%}", "середня": f"{v.mean():+.2%}", "PF": pf(v), "PF без топ-3": pf(v.iloc[:-3]),
                     "PF ≥ 20 млн": pf(df[df["band"] == "≥ 20 млн"]["ret"]), "PF 5–20 млн": pf(df[df["band"] == "5–20 млн"]["ret"]),
                     "пропущено ≥ +10%": f"{(~big.index.isin(got.index)).sum()} з {len(big)}",
                     "очікування, хв": round(df["wait"].mean() * 15) if len(df) else None}
    pd.set_option("display.width", 260)
    print(f"{len(coins)} монет поза навчанням; вихід — частковий (½ на +3 ATR, далі беззбиток), стоп {k:.2f} ATR\n")
    print(pd.DataFrame(res).T.to_string())


if __name__ == "__main__":
    main()
