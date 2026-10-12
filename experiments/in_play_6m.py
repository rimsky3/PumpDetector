"""
experiments/in_play_6m.py — МОНЕТА «В ГРІ»: чи кращі угоди моделі на таких монетах (2026-10-12; власник: «як визначити тікер,
який in play, і прогнати модель тільки на них?»). Повтор питання in_play.py (01.10: на 30 днях і старій моделі — різниці
немає, модель і так входить майже лише на «гарячих»), тепер на 6 міс., робочих моделях (split і market) і з визначеннями,
яких модель НЕ бачить: місце монети серед усього ринку (ознаки моделі відносні лише до власної історії монети).

Визначення (зафіксовано ДО запуску; усе — на момент входу, з 15m-цін кешу data/long_bin; «звичайний оборот» — медіана
денного обороту за 30 повних днів до дня входу):
    hot        оборот за 24 год до входу ≥ 3 × звичайного
    days       обидва попередні ПОВНІ дні оборот ≥ 2 × звичайного (у грі вже кілька днів, не лише зараз)
    leader     монета в топ-10% усіх монет кешу за (оборот 24 год ÷ звичайний) у цей момент
    strong     монета в топ-10% усіх монет за зміною ціни за 3 дні (72 год) у цей момент
КРИТЕРІЙ для кожного: PF(в грі) − PF(решта) ≥ +0.10 у split на ≥ 20 млн; той самий знак в обох половинах монет; краще в
≥ 4 з 6 місяців; той самий знак у market. Окремо — частка угод «в грі»: якщо вона ~90%, фільтр нічого не відсіює.
Угоди — робочі моделі за 6 міс. (reports/candidate_6m_trades.csv, набір current; спершу candidate_6m.py).
→ reports/in_play_6m.md

    python experiments/in_play_6m.py
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIR = os.path.join("data", "long_bin")
TRADES = os.path.join("reports", "candidate_6m_trades.csv")
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "in_play_6m.md")
DEFS = {"hot": "Гарячий обсяг (24 год ≥ 3× звичайного)", "days": "У грі вже 2 дні (обидва ≥ 2× звичайного)",
        "leader": "Лідер ринку (топ-10% за ростом обороту)", "strong": "Сильніша за ринок (топ-10% за ростом ціни за 3 дні)"}


def pf(r) -> float:
    r = np.asarray(r, dtype=float)
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 and len(r) else float("nan")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Монета «в грі» (4 визначення) проти результату угод робочих моделей, 6 міс.").parse_args()
    ratio, ret3, days2 = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(DIR, "*.pkl"))):
        c = os.path.basename(f)[:-4]
        h = pickle.load(open(f, "rb"))["h"]
        if h.empty:
            continue
        day = h["volume"].resample("1D").sum()
        norm_d = day.rolling(30, min_periods=10).median().shift(1)               # звичайний оборот: 30 повних днів до сьогодні
        norm = pd.Series(norm_d.reindex(h.index.floor("1D")).to_numpy(), index=h.index)
        ratio[c] = (h["volume"].rolling(96, min_periods=48).sum() / norm).astype("float32")
        ret3[c] = (h["price"] / h["price"].shift(288) - 1).astype("float32")
        hot_d = day / norm_d >= 2
        both = (hot_d.shift(1) & hot_d.shift(2)).astype(float)                    # учора й позавчора
        days2[c] = pd.Series(both.reindex(h.index.floor("1D")).to_numpy(), index=h.index)
    R, P3 = pd.DataFrame(ratio), pd.DataFrame(ret3)
    lead = R.rank(axis=1, pct=True)
    strong = P3.rank(axis=1, pct=True)
    print(f"монет у кеші {R.shape[1]}, барів {R.shape[0]}", flush=True)

    tr = pd.read_csv(TRADES, parse_dates=["ts"])
    tr = tr[tr["set"] == "current"].copy()
    rows = []
    for c, g in tr.groupby("ticker"):
        if c not in R:
            continue
        i = R.index.get_indexer(g["ts"])
        ok = i >= 0
        g = g[ok].copy()
        i = i[ok]
        g["hot"] = R[c].to_numpy()[i] >= 3
        g["days"] = days2[c].reindex(R.index).to_numpy()[i] >= 1
        g["leader"] = lead[c].to_numpy()[i] >= 0.9
        g["strong"] = strong[c].to_numpy()[i] >= 0.9
        g["ratio"] = R[c].to_numpy()[i]
        rows.append(g)
    d = pd.concat(rows).dropna(subset=["ratio"])
    big = d[d["band"] == BIG]
    L = ["# Монета «в грі» проти результату угод моделі, 6 місяців (12.10)", "",
         "Угоди робочих моделей (набір current), монети ≥ 20 млн; стан монети — на момент входу.", "",
         "| Визначення | Режим | Частка угод «в грі» | В грі: угод · PF · медіана · стоп | Решта: угод · PF · медіана · стоп | Різниця PF |",
         "|---|---|---|---|---|---|"]
    notes, verdict = [], {}
    for key, title in DEFS.items():
        res = {}
        for mode in ("split", "market"):
            g = big[big["mode"] == mode]
            a, b = g[g[key]], g[~g[key]]
            cell = lambda x: f"{len(x)} · {pf(x['ret']):.2f} · {x['ret'].median():+.2%} · {(x['reason'] == 'stop').mean():.0%}"     # noqa: E731
            diff = pf(a["ret"]) - pf(b["ret"])
            halves = [pf(a[a["half"] == hf]["ret"]) - pf(b[b["half"] == hf]["ret"]) for hf in ("A", "B")]
            months = [pf(a[a["month"] == m]["ret"]) - pf(b[b["month"] == m]["ret"]) for m in sorted(g["month"].unique())]
            res[mode] = (diff, halves, months)
            L.append(f"| {title} | {mode} | {len(a) / max(1, len(g)):.0%} | {cell(a)} | {cell(b)} | **{diff:+.2f}** |")
        (ds, hs, ms), (dm, _, mm) = res["split"], res["market"]
        ok = ds >= 0.10 and all(x > 0 for x in hs) and sum(x > 0 for x in ms) >= 4 and dm > 0
        verdict[key] = ok
        notes.append(f"- **{title}**: split {ds:+.2f}; половини A / B {hs[0]:+.2f} / {hs[1]:+.2f}; по місяцях "
                     + " / ".join(f"{x:+.2f}" for x in ms) + f" → краще в {sum(x > 0 for x in ms)} з {len(ms)}; market {dm:+.2f} "
                     f"(краще в {sum(x > 0 for x in mm)} з {len(mm)}) → **{'ВИКОНАНО' if ok else 'НЕ ВИКОНАНО'}**")
    L += ["", "## Критерій", ""] + notes
    q = big[big["mode"] == "split"]["ratio"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    L += ["", f"Оборот за 24 год до входу відносно звичайного в угодах split: 10% — ×{q.iloc[0]:.1f}, 25% — ×{q.iloc[1]:.1f}, "
          f"медіана ×{q.iloc[2]:.1f}, 75% — ×{q.iloc[3]:.1f}, 90% — ×{q.iloc[4]:.1f}."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
