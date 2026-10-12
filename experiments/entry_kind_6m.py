"""
experiments/entry_kind_6m.py — ТИП ВХОДУ на 6 місяцях: з відкату чи на хаях, «мертва зона» (2026-10-12; власник: RSR 12.10 —
«цей вхід хороший, бо не на імпульсі»; розвідка experiments/entry_kind.py на 30 днях дала гіпотези, тут — перевірка).

Для кожної угоди на барі входу (закриття бару ts) з 15m-цін кешу data/long_bin:
    from_hi = ціна / хай 24 год − 1;  from_lo = ціна / лоу 24 год − 1;  run4h = ціна / ціна 4 год тому − 1.
Групи (пороги — ті самі, що в розвідці, не підбирались на цих даних):
    гаряча + відкат   from_lo ≥ +10% і from_hi ≤ −3%        гаряча + на хаях   from_lo ≥ +10% і from_hi > −1%
    мертва зона       +3% ≤ run4h < +8%                      решта              run4h < +3% або ≥ +8%

КРИТЕРІЙ (до запуску), монети ≥ 20 млн, основний набір — угоди split_start30 за 6 міс. (reports/early_target_trades.csv):
    H1  PF(гаряча + відкат) − PF(гаряча + на хаях) ≥ +0.10 загалом і різниця > 0 у ≥ 4 з 6 місяців;
    H2  PF(мертва зона) − PF(решта) ≤ −0.10 загалом і різниця < 0 у ≥ 4 з 6 місяців.
Стійкість (без окремого критерію): ті самі групи на угодах base / market — 5 seeds (reports/time_seeds_trades.csv) і на
purged-прогоні split, де тестовий місяць вирізано з навчання (reports/purged_split_trades.csv; жовтень — без цін у кеші).
→ reports/entry_kind_6m.md

    python experiments/entry_kind_6m.py            # лише файли, без бази; ~1 хв
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DIR = os.path.join("data", "long_bin")
BIG = "≥ 20 млн"
OUT = os.path.join("reports", "entry_kind_6m.md")
SETS = [("split, 6 міс. (основний)", "reports/early_target_trades.csv", "split_start30"),
        ("market, 6 міс., 5 seeds", "reports/time_seeds_trades.csv", "base"),
        ("split, purged (серп–вер)", "reports/purged_split_trades.csv", "split")]


def pf(r) -> float:
    r = np.asarray(r, dtype=float)
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 and len(r) else float("nan")


def context(tr: pd.DataFrame) -> pd.DataFrame:
    """Додає from_hi, from_lo, run4h на барі входу; угоди без цін у кеші відкидаються."""
    out = []
    for c, g in tr.groupby("ticker"):
        f = os.path.join(DIR, f"{c}.pkl")
        if not os.path.exists(f):
            continue
        h = pickle.load(open(f, "rb"))["h"]
        price = h["price"]
        hi = h["high"].rolling(96, min_periods=48).max()
        lo = h["low"].rolling(96, min_periods=48).min()
        i = h.index.get_indexer(g["ts"])
        ok = i >= 0
        g = g[ok].copy()
        i = i[ok]
        p = price.to_numpy()[i]
        g["from_hi"] = p / hi.to_numpy()[i] - 1
        g["from_lo"] = p / lo.to_numpy()[i] - 1
        g["run4h"] = p / price.shift(16).to_numpy()[i] - 1
        out.append(g)
    d = pd.concat(out) if out else tr.iloc[:0]
    return d.dropna(subset=["from_hi", "from_lo", "run4h"])


def groups(d: pd.DataFrame) -> dict:
    hot = d["from_lo"] >= 0.10
    dead = (d["run4h"] >= 0.03) & (d["run4h"] < 0.08)
    return {"гаряча + відкат": hot & (d["from_hi"] <= -0.03), "гаряча + на хаях": hot & (d["from_hi"] > -0.01),
            "мертва зона (+3…+8% за 4 год)": dead, "решта": ~dead,
            "до імпульсу (< +3% за 4 год)": d["run4h"] < 0.03, "на імпульсі (≥ +8% за 4 год)": d["run4h"] >= 0.08}


def pair(d: pd.DataFrame, a: str, b: str) -> tuple[float, list[float]]:
    g = groups(d)
    total = pf(d[g[a]]["ret"]) - pf(d[g[b]]["ret"])
    by = []
    for m, x in d.groupby("month"):
        gx = groups(x)
        by.append(pf(x[gx[a]]["ret"]) - pf(x[gx[b]]["ret"]))
    return total, by


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Тип входу на 6 міс.: з відкату чи на хаях, «мертва зона» +3…+8% за 4 год").parse_args()
    L = ["# Тип входу на 6 місяцях: з відкату чи на хаях (12.10)", "",
         "На барі входу: відстань від хаю / лоу 24 год і ріст за 4 год. Пороги — з розвідки на 30 днях (experiments/entry_kind.py). "
         "Монети ≥ 20 млн, якщо не сказано інше.", ""]
    verdict = {}
    for title, path, variant in SETS:
        tr = pd.read_csv(path, parse_dates=["ts"])
        tr = tr[tr["variant"] == variant]
        tr["month"] = tr["ts"].dt.strftime("%Y-%m")
        d = context(tr)
        big = d[d["band"] == BIG]
        L += [f"## {title}", "", f"Угод {len(tr)}, з цінами в кеші {len(d)}, з них ≥ 20 млн {len(big)}; PF ≥ 20 млн {pf(big['ret']):.2f}, усі {pf(d['ret']):.2f}.", "",
              "| Група | Угод | PF | Медіана | У плюсі | По стопу | PF усі монети |", "|---|---|---|---|---|---|---|"]
        gb, ga = groups(big), groups(d)
        for name in gb:
            x = big[gb[name]]
            L.append(f"| {name} | {len(x)} | {pf(x['ret']):.2f} | {x['ret'].median():+.2%} | {(x['ret'] > 0).mean():.0%} | "
                     f"{(x['reason'] == 'stop').mean():.0%} | {pf(d[ga[name]]['ret']):.2f} |")
        t1, m1 = pair(big, "гаряча + відкат", "гаряча + на хаях")
        t2, m2 = pair(big, "мертва зона (+3…+8% за 4 год)", "решта")
        months = sorted(big["month"].unique())
        L += ["", f"- відкат − хаї: {t1:+.2f} загалом; по місяцях ({', '.join(m[5:] for m in months)}): " + " / ".join(f"{q:+.2f}" for q in m1) +
              f" → краще в {sum(q > 0 for q in m1)} з {len(m1)}",
              f"- мертва зона − решта: {t2:+.2f} загалом; по місяцях: " + " / ".join(f"{q:+.2f}" for q in m2) +
              f" → гірше в {sum(q < 0 for q in m2)} з {len(m2)}", ""]
        verdict[title] = (t1, m1, t2, m2)
    t1, m1, t2, m2 = verdict[SETS[0][0]]
    need = 4 if len(m1) >= 6 else int(np.ceil(len(m1) * 2 / 3))
    h1 = t1 >= 0.10 and sum(q > 0 for q in m1) >= need
    h2 = t2 <= -0.10 and sum(q < 0 for q in m2) >= need
    L += ["## Критерій (основний набір — split, 6 міс., ≥ 20 млн)", "",
          f"- **H1 (відкат кращий за хаї на ≥ 0.10 і в ≥ {need} з {len(m1)} місяців): {'ВИКОНАНО' if h1 else 'НЕ ВИКОНАНО'}** ({t1:+.2f}, {sum(q > 0 for q in m1)} з {len(m1)})",
          f"- **H2 (мертва зона гірша за решту на ≥ 0.10 і в ≥ {need} з {len(m2)} місяців): {'ВИКОНАНО' if h2 else 'НЕ ВИКОНАНО'}** ({t2:+.2f}, {sum(q < 0 for q in m2)} з {len(m2)})"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
