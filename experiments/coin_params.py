"""
experiments/coin_params.py — ПАРАМЕТРИ МОНЕТИ, на які дивиться власник, проти результату угод моделі (2026-10-12).
Власник: «дивлюсь на ліквідність у стакані; кількість трейдів — стрічка не повинна бути мертва; як виріс OI; чи давала монета
імпульси раніше». Питання: чи угоди моделі кращі на монетах, де ці параметри вищі? Якщо так — це фільтр монети / підказка в
списку. Розбивка тих самих 30 днів за оборотом і OI виглядала сильно, але там параметри рахувались за період самого бектесту
(погляд уперед) — тут усе виміряно ДО входу.

Параметри на барі входу (ts — відкриття бару, вхід на його закритті), з кешу data/long_bin і data/trades15m:
    depth      глибина стакану ±2% від ціни, $ (біди + аски; з book_depth2_turn × оборот 24 год), медіана за 4 год до входу
    trades     кількість угод Binance за 24 год до входу
    oi_7d      зміна OI за 7 днів до входу (ланцюжок oi_chg_24h)
    hist       скільки днів за 30 днів (без останньої доби) монета давала стрибок ≥ +15% (хай дня до закриття попереднього)
Очікування власника для всіх чотирьох: більше — краще.

КРИТЕРІЙ (до запуску), для кожного параметра окремо. Угоди РОБОЧИХ моделей за 6 міс. (reports/candidate_6m_trades.csv, набір
current), монети ≥ 20 млн, основний режим split; третини — за значенням параметра серед цих угод:
    PF верхньої третини − PF нижньої ≥ +0.10; той самий знак в обох половинах монет (md5); верхня краща за нижню в ≥ 4 з 6
    місяців; і той самий знак різниці в режимі market (закономірність лише в одному наборі угод — шум: урок entry_kind_6m).
→ reports/coin_params.md

    python experiments/coin_params.py             # спершу experiments/candidate_6m.py (пише угоди з часом входу)
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DIR, TR_DIR = os.path.join("data", "long_bin"), os.path.join("data", "trades15m")
TRADES = os.path.join("reports", "candidate_6m_trades.csv")
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "coin_params.md")
PARAMS = {"depth": "Глибина стакану ±2%, $", "trades": "Угод за 24 год", "oi_7d": "Зміна OI за 7 днів", "hist": "Днів зі стрибком ≥ +15% за 30 дн"}
FMT = {"depth": lambda v: f"{v / 1e6:.2f} млн", "trades": lambda v: f"{v / 1e3:.0f} тис.", "oi_7d": lambda v: f"{v:+.0%}", "hist": lambda v: f"{v:.0f}"}


def pf(r) -> float:
    r = np.asarray(r, dtype=float)
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 and len(r) else float("nan")


def coin_frame(c: str) -> pd.DataFrame | None:
    """Параметри монети на кожному 15m-барі (індекс — відкриття бару); значення бару знають лише минуле до його закриття."""
    f = os.path.join(DIR, f"{c}.pkl")
    if not os.path.exists(f):
        return None
    x = pickle.load(open(f, "rb"))
    h, f15 = x["h"], x["f15"].set_index("ts")
    out = pd.DataFrame(index=h.index)
    turn24 = h["volume"].rolling(96, min_periods=48).sum()
    if "book_depth2_turn" in f15:
        depth = np.exp(f15["book_depth2_turn"].reindex(h.index)) * turn24
        out["depth"] = depth.rolling(16, min_periods=4).median()
    oi = np.log1p(f15["oi_chg_24h"].reindex(h.index))
    out["oi_7d"] = np.expm1(sum(oi.shift(96 * k) for k in range(7)))
    tf = os.path.join(TR_DIR, f"{c}.parquet")
    if os.path.exists(tf):
        n = pd.read_parquet(tf)["n"]
        n.index = pd.to_datetime(n.index, utc=True)
        out["trades"] = n.reindex(h.index).rolling(96, min_periods=48).sum()
    day = h.resample("1D").agg({"high": "max", "price": "last"})
    jump = (day["high"] / day["price"].shift(1) - 1 >= 0.15).astype(float)
    hist = jump.rolling(30, min_periods=10).sum().shift(1)                 # 30 днів, що закінчились учора
    out["hist"] = hist.reindex(h.index.floor("1D")).to_numpy()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Параметри монети (стакан, угоди, OI, історія імпульсів) проти результату угод моделі, 6 міс.").parse_args()
    tr = pd.read_csv(TRADES, parse_dates=["ts"])
    tr = tr[tr["set"] == "current"].copy()
    parts = []
    for c, g in tr.groupby("ticker"):
        cf = coin_frame(c)
        if cf is None:
            continue
        i = cf.index.get_indexer(g["ts"])
        ok = i >= 0
        g = g[ok].copy()
        for col in PARAMS:
            g[col] = cf[col].to_numpy()[i[ok]] if col in cf else np.nan
        parts.append(g)
    d = pd.concat(parts)
    big = d[d["band"] == BIG]
    L = ["# Параметри монети проти результату угод моделі, 6 місяців (12.10)", "",
         "Угоди робочих моделей (набір current з candidate_6m), монети ≥ 20 млн. Параметри виміряно до входу. Третини — за значенням "
         "параметра серед угод цього режиму. Очікування власника: більше — краще.", ""]
    verdict = {}
    for col, title in PARAMS.items():
        L += [f"## {title}", "", "| Режим | Третина | Діапазон | Угод | Монет | PF | Медіана | У плюсі | По стопу |", "|---|---|---|---|---|---|---|---|---|"]
        res = {}
        for mode in ("split", "market"):
            g = big[(big["mode"] == mode)].dropna(subset=[col])
            cov = len(g) / max(1, len(big[big["mode"] == mode]))
            if col == "hist":                               # мало різних значень — групи 0 / 1–2 / 3+ замість третин
                q = pd.cut(g[col], [-1, 0, 2, 1e9], labels=["низ", "середина", "верх"])
            else:
                q = pd.qcut(g[col].rank(method="first"), 3, labels=["низ", "середина", "верх"])
            g = g.assign(q=q)
            for name in ("низ", "середина", "верх"):
                x = g[g["q"] == name]
                rng = f"{FMT[col](x[col].min())} … {FMT[col](x[col].max())}" if len(x) else "—"
                L.append(f"| {mode} | {name} | {rng} | {len(x)} | {x['ticker'].nunique()} | {pf(x['ret']):.2f} | {x['ret'].median():+.2%} | "
                         f"{(x['ret'] > 0).mean():.0%} | {(x['reason'] == 'stop').mean():.0%} |")
            top, bot = g[g["q"] == "верх"], g[g["q"] == "низ"]
            diff = pf(top["ret"]) - pf(bot["ret"])
            halves = [pf(top[top["half"] == hf]["ret"]) - pf(bot[bot["half"] == hf]["ret"]) for hf in ("A", "B")]
            months = [pf(top[top["month"] == m]["ret"]) - pf(bot[bot["month"] == m]["ret"]) for m in sorted(g["month"].unique())]
            res[mode] = (diff, halves, months, cov)
        (ds, hs, ms, cs), (dm, _, mm, _) = res["split"], res["market"]
        ok = ds >= 0.10 and all(x > 0 for x in hs) and sum(x > 0 for x in ms) >= 4 and dm > 0
        verdict[col] = ok
        L += ["", f"- split: верх − низ **{ds:+.2f}**; половини монет A / B {hs[0]:+.2f} / {hs[1]:+.2f}; по місяцях "
              + " / ".join(f"{x:+.2f}" for x in ms) + f" → краще в {sum(x > 0 for x in ms)} з {len(ms)}; покриття даними {cs:.0%}",
              f"- market: верх − низ {dm:+.2f}; краще в {sum(x > 0 for x in mm)} з {len(mm)} місяців",
              f"- **Критерій: {'ВИКОНАНО' if ok else 'НЕ ВИКОНАНО'}**", ""]
    L += ["## Підсумок", ""] + [f"- {PARAMS[c]}: {'підтверджено' if v else 'не підтверджено'}" for c, v in verdict.items()]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
