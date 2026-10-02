"""
experiments/exit_structure.py — правила виходу від власника (2026-10-01) на монетах бектесту (поза навчанням).

Розбір двох найбільших стопів дня:
    MOVR 12:15 — сигнал на хаях рейнджу без пробою, далі кожна 5-хвилинка червона, OI виходить; стоп k·ATR під лоу
                 рейнджу вибили (−14%). Власник: «сенсу чекати стоп немає» → РАННІЙ ВИХІД без продовження.
    US 17:45   — «типовий підбір»: стоп треба одразу за лоу свічки входу, а не k·ATR (на волатильній монеті −14%)
                 → СТОП ЗА СТРУКТУРОЮ.

Логіка позиції — як risk.position_state (вхід на першому барі Alert, вихід — 4 бари без Alert або стоп; після виходу
новий вхід, щойно Alert знову горить), змінюється лише вихід:

    A  як зараз            стоп entry − k·ATR(15m)
    B2 / B3                + ранній вихід: n (2 / 3) барів поспіль закрились червоними (close < open), OI падав на кожному,
                             і ціна за ці бари не оновила максимум з моменту входу → вихід за закриттям
    C  стоп за свічкою     стоп = лоу бару входу − 0.1% (замість k·ATR)
    D2 / D3                C + ранній вихід

Стоп перевіряється раніше за інші виходи (консервативно: усередині бару спершу лоу). Комісія 0.1% за угоду.
Метрики: угоди, % у плюсі, середня, медіана, сума, сума без топ-3, профіт-фактор, % по стопу, утримання;
«зрізані виграші» — угоди варіанта A з результатом ≥ +5%, що у варіанті з тим самим входом дали < +1%.

    python experiments/exit_structure.py                 # монети двох основних прогонів бектесту
    python experiments/exit_structure.py --refresh       # перерахувати оцінки моделі (інакше — кеш)
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from backtest import tiers_and_bars
from risk import atr

FEE = 0.001
BAR_TP, BAR_SL, BAR_H = 3.0, 1.5, 16          # вихід «бар'єри»: +3 ATR / −1.5 ATR / 4 год (як ціль у outcome_target.py)
RUNS = ["20261001_1441_current_alert", "20261001_1517_current_alert"]
CACHE = os.path.join("reports", "backtests", "_exit_structure_cache.pkl")


def simulate(h: pd.DataFrame, score: pd.Series, k: float, exit_bars: int, stop_mode: str, early: int) -> list[dict]:
    a = atr(h)
    score = score.reindex(h.index)
    o, hi, lo, c, oi = (h[x].to_numpy() for x in ("open", "high", "low", "price", "oi"))
    idx, sc, av = h.index, score.to_numpy(), a.reindex(h.index).to_numpy()
    out, tr, below, red = [], None, 0, 0
    for i in range(len(idx)):
        p = sc[i]
        if np.isnan(p):
            continue
        if tr is None:
            if p >= 1 and not np.isnan(c[i]) and not np.isnan(av[i]):
                stop = c[i] - k * av[i] if stop_mode == "atr" else (c[i] - BAR_SL * av[i] if stop_mode == "barrier" else lo[i] * (1 - 0.001))
                tr = {"i0": i, "entry_ts": idx[i], "entry": c[i], "stop": stop, "high": hi[i], "tp": c[i] + BAR_TP * av[i]}
                below, red = 0, 0
            continue
        exit_px, reason = None, None
        if lo[i] <= tr["stop"]:
            exit_px, reason = tr["stop"], "stop"
        elif stop_mode == "barrier":               # бар'єри: ціль або час; сигнали моделі на вихід не впливають
            if hi[i] >= tr["tp"]:
                exit_px, reason = tr["tp"], "target"
            elif i - tr["i0"] >= BAR_H:
                exit_px, reason = c[i], "time"
        else:
            new_high = hi[i] > tr["high"]
            is_red = c[i] < o[i] and i > 0 and not np.isnan(oi[i]) and not np.isnan(oi[i - 1]) and oi[i] < oi[i - 1]
            red = red + 1 if (is_red and not new_high) else 0
            below = below + 1 if p < 1 else 0
            if early and red >= early:
                exit_px, reason = c[i], "early"
            elif below >= exit_bars:
                exit_px, reason = c[i], "model"
        tr["high"] = max(tr["high"], hi[i])
        if exit_px is not None:
            out.append({"entry_ts": tr["entry_ts"], "exit_ts": idx[i], "ret": exit_px / tr["entry"] - 1 - FEE,
                        "reason": reason, "bars": i - tr["i0"]})
            tr = None
    return out


def stats(r: pd.DataFrame) -> dict:
    v = r["ret"].sort_values()
    if v.empty:
        return {}
    gw, gl = v[v > 0].sum(), -v[v <= 0].sum()
    return {"угод": len(v), "у плюсі": f"{(v > 0).mean():.0%}", "сер.": f"{v.mean():+.2%}", "медіана": f"{v.median():+.2%}",
            "сума": f"{v.sum():+.0%}", "без топ-3": f"{v.iloc[:-3].sum():+.0%}", "PF": f"{gw / gl:.2f}" if gl else "∞",
            "сер. програш": f"{v[v <= 0].mean():+.2%}", "по стопу": f"{(r['reason'] == 'stop').mean():.0%}",
            "рано": f"{(r['reason'] == 'early').mean():.0%}", "утрим., год": f"{r['bars'].median() / 4:.1f}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Стоп за свічкою входу і ранній вихід без продовження (експеримент)")
    ap.add_argument("--refresh", action="store_true", help="перерахувати оцінки моделі на монетах (інакше кеш)")
    ap.add_argument("--days", type=int, default=30, help="днів історії (як у бектесті)")
    args = ap.parse_args()
    load_dotenv(".env")
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    k, exit_bars = m15.get("risk", {}).get("k"), m15.get("exit_bars", 4)
    band = {}
    for run in RUNS:
        d = json.load(open(os.path.join("reports", "backtests", run + ".json"), encoding="utf-8"))
        for u in d["meta"]["universe"]:
            band[u["base"]] = "≥ 20 млн" if d["meta"]["min_turnover"] >= 20e6 else "5–20 млн"
    if os.path.exists(CACHE) and not args.refresh:
        data = pickle.load(open(CACHE, "rb"))
    else:
        data = {}
        with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            for n, b in enumerate(sorted(band)):
                th = tiers_and_bars(conn, b, m15, m1, args.days)
                if th is not None:
                    data[b] = (th[0][["alert_score"]], th[1][["open", "high", "low", "price", "oi"]])
                if n % 20 == 0:
                    print(f"  оцінено {n + 1}/{len(band)}", flush=True)
        pickle.dump(data, open(CACHE, "wb"))
    print(f"монет: {len(data)}; k = {k:.2f} ATR, вихід моделі — {exit_bars} бари\n")

    variants = {"A  як зараз (k·ATR)": ("atr", 0), "B2 + ранній вихід (2 бари)": ("atr", 2), "B3 + ранній вихід (3 бари)": ("atr", 3),
                "C  стоп за свічкою": ("candle", 0), "D2 C + ранній (2)": ("candle", 2), "D3 C + ранній (3)": ("candle", 3),
                "E  бар'єри +3/−1.5 ATR, 4 год": ("barrier", 0)}
    res = {}
    for name, (sm, early) in variants.items():
        rows = []
        for b, (t, h) in data.items():
            for x in simulate(h, t["alert_score"], k, exit_bars, sm, early):
                rows.append({**x, "ticker": b, "band": band[b]})
        res[name] = pd.DataFrame(rows)
    base = res["A  як зараз (k·ATR)"].set_index(["ticker", "entry_ts"])["ret"]
    table = {}
    for name, r in res.items():
        s = stats(r)
        m = r.set_index(["ticker", "entry_ts"])["ret"]
        won = base[base >= 0.05]
        same = m.reindex(won.index).dropna()
        s["зрізані виграші"] = f"{int((same < 0.01).sum())} з {len(won)}" if not name.startswith("A") else f"— ({len(won)} ≥ +5%)"
        table[name] = s
        for bnd, g in r.groupby("band"):
            table[f"    {name.split()[0]} · {bnd}"] = stats(g)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(table).T.fillna("").to_string())
    for name, r in res.items():
        r.to_csv(os.path.join("reports", f"exit_structure_{name.split()[0]}.csv"), index=False)


if __name__ == "__main__":
    main()
