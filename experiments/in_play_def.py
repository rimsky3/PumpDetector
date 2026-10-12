"""
experiments/in_play_def.py — ВИЗНАЧЕННЯ «В ГРІ» З ПРИКЛАДІВ ВЛАСНИКА на 6 місяцях (2026-10-12).
Власник дав приклади MAGIC 09.10 і OGN 08.10; experiments/in_play_examples.py показав, чим вони виділяються серед звичайних
входів моделі: тиха монета (звичайний оборот 1–4 млн $), оборот у десятки разів вищий за звичайний, OI за добу росте в рази,
це триває багато годин, відкати неглибокі, OI не падає. Визначення зафіксовано за ПРИКЛАДАМИ (пороги — найслабші значення
обох прикладів у перші 12 год), до перегляду результатів угод:

    active   оборот за 24 год ≥ 8 × звичайного  І  OI за 24 год ≥ +45%  І  ≥ 8 «гарячих» годин за добу (погодинний оборот
             ≥ 3 × звичайного)  І  ціна за 24 год ≥ +15%
    alive    OI не нижче ніж −10% від свого максимуму за 12 год  І  найглибший відкат за 12 год ≤ ½ руху від лоу 24 год
    В ГРІ = active і alive

Застереження: приклади — це рухи, які ВДАЛИСЯ; визначення перевіряє, чи схожі за станом моменти в середньому кращі.
Чесний список 6 міс. містить монети із сер. оборотом ≥ 5 млн, тож найтихіші (як OGN, 1 млн) сюди не входять — для них є
experiments/wake_channel.py. КРИТЕРІЙ (до запуску), усі монети списку: PF(в грі) − PF(решта) ≥ +0.10 у split; той самий знак
в обох половинах монет; краще в ≥ 4 з 6 місяців; той самий знак у market. Якщо угод «в грі» < 100 — «даних замало».
Угоди — робочі моделі (reports/candidate_6m_trades.csv, набір current). → reports/in_play_def.md

    python experiments/in_play_def.py
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIR = os.path.join("data", "long_bin")
TRADES = os.path.join("reports", "candidate_6m_trades.csv")
OUT = os.path.join("reports", "in_play_def.md")


def pf(r) -> float:
    r = np.asarray(r, dtype=float)
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 and len(r) else float("nan")


def state(c: str) -> pd.DataFrame | None:
    f = os.path.join(DIR, f"{c}.pkl")
    if not os.path.exists(f):
        return None
    x = pickle.load(open(f, "rb"))
    h, f15 = x["h"], x["f15"].set_index("ts")
    v, p = h["volume"], h["price"]
    usual_d = v.resample("1D").sum().rolling(30, min_periods=7).median().shift(1)
    usual = pd.Series(usual_d.reindex(h.index.floor("1D")).to_numpy(), index=h.index)
    out = pd.DataFrame(index=h.index)
    out["vol24_x"] = v.rolling(96, min_periods=48).sum() / usual
    out["hot_hours"] = (v.rolling(4, min_periods=4).sum() >= 3 * usual / 24).rolling(96, min_periods=24).sum() / 4
    out["oi_24h"] = f15["oi_chg_24h"].reindex(h.index)
    out["ret_24h"] = p / p.shift(96) - 1
    oi_idx = np.log1p(f15["oi_chg_bar"].reindex(h.index)).fillna(0).cumsum()        # лог-індекс OI з побарових змін
    out["oi_hold"] = np.expm1(oi_idx - oi_idx.rolling(48, min_periods=8).max())
    hi24, lo24 = h["high"].rolling(96, min_periods=24).max(), h["low"].rolling(96, min_periods=24).min()
    dd = (h["high"].rolling(48, min_periods=8).max() - h["low"]).rolling(48, min_periods=8).max()
    out["pull_12h"] = dd / (hi24 - lo24).where(hi24 > lo24)
    out["usual"] = usual / 1e6
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Визначення «в грі» з прикладів власника (MAGIC, OGN) проти результату угод, 6 міс.").parse_args()
    tr = pd.read_csv(TRADES, parse_dates=["ts"])
    tr = tr[tr["set"] == "current"].copy()
    parts = []
    for c, g in tr.groupby("ticker"):
        s = state(c)
        if s is None:
            continue
        i = s.index.get_indexer(g["ts"])
        ok = i >= 0
        g = g[ok].copy()
        for col in s.columns:
            g[col] = s[col].to_numpy()[i[ok]]
        parts.append(g)
    d = pd.concat(parts).dropna(subset=["vol24_x", "oi_24h", "ret_24h"])
    d["active"] = (d["vol24_x"] >= 8) & (d["oi_24h"] >= 0.45) & (d["hot_hours"] >= 8) & (d["ret_24h"] >= 0.15)
    d["alive"] = (d["oi_hold"] >= -0.10) & (d["pull_12h"] <= 0.5)
    d["inplay"] = d["active"] & d["alive"]
    L = ["# Визначення «в грі» з прикладів власника, 6 місяців (12.10)", "",
         "В ГРІ = оборот 24 год ≥ 8× звичайного, OI за 24 год ≥ +45%, ≥ 8 гарячих годин, ціна за 24 год ≥ +15% (active) "
         "і OI не нижче −10% від максимуму за 12 год, відкат ≤ ½ руху (alive). Угоди робочих моделей, усі монети чесного списку.", "",
         "| Режим | Група | Угод | Монет | PF | Медіана | У плюсі | По стопу | Сер. звичайний оборот, млн $ |", "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for mode in ("split", "market"):
        g = d[d["mode"] == mode]
        groups = {"В ГРІ (active і alive)": g[g["inplay"]], "active, але не alive": g[g["active"] & ~g["alive"]], "решта": g[~g["active"]]}
        for name, x in groups.items():
            L.append(f"| {mode} | {name} | {len(x)} | {x['ticker'].nunique()} | {pf(x['ret']):.2f} | {x['ret'].median():+.2%} | "
                     f"{(x['ret'] > 0).mean():.0%} | {(x['reason'] == 'stop').mean():.0%} | {x['usual'].median():.1f} |")
        a, b = g[g["inplay"]], g[~g["inplay"]]
        diff = pf(a["ret"]) - pf(b["ret"])
        halves = [pf(a[a["half"] == hf]["ret"]) - pf(b[b["half"] == hf]["ret"]) for hf in ("A", "B")]
        months = [(m, len(a[a["month"] == m]), pf(a[a["month"] == m]["ret"]) - pf(b[b["month"] == m]["ret"])) for m in sorted(g["month"].unique())]
        res[mode] = (len(a), diff, halves, months)
    (n, ds, hs, ms), (_, dm, _, mm) = res["split"], res["market"]
    better = sum(x > 0 for _, k, x in ms if k > 0)
    enough = n >= 100
    ok = enough and ds >= 0.10 and all(x > 0 for x in hs) and better >= 4 and dm > 0
    L += ["", f"- split: в грі − решта **{ds:+.2f}** ({n} угод «в грі»); половини A / B {hs[0]:+.2f} / {hs[1]:+.2f}; по місяцях (угод · різниця): "
          + ", ".join(f"{m[5:]}: {k} · {x:+.2f}" for m, k, x in ms) + f" → краще в {better} з {len(ms)}",
          f"- market: в грі − решта {dm:+.2f} ({res['market'][0]} угод); по місяцях: " + ", ".join(f"{m[5:]}: {k} · {x:+.2f}" for m, k, x in mm),
          f"- **Критерій: {'ВИКОНАНО' if ok else ('ДАНИХ ЗАМАЛО' if not enough else 'НЕ ВИКОНАНО')}**"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
