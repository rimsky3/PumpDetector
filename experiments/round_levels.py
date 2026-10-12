"""
experiments/round_levels.py — КРУГЛІ ЦІНОВІ РІВНІ: чи реагує на них ціна частіше, ніж на випадкові (2026-10-13).
Власник: «а якщо ознаками зробити круглі числа? цінові рівні — дуже часто ціна на них реагує». Перш ніж робити ознаки —
перевірка самої реакції, тим самим тестом, що й рівні профілю обсягу 03.10 (experiments/profile_levels.py: відскок чи
прохід після першого дотику; POC / VAH / VAL тоді вийшли ≈ випадковим рівням).

Круглий рівень — за мантисою ціни (ціна = m × 10^k, 1 ≤ m < 10), тож визначення однакове для 0.0025 і 250:
    великі      m ∈ {1, 2, 5}                 (0.001, 0.02, 0.5, 1, 2, 5, 10 …)
    цілі        m ∈ {3, 4, 6, 7, 8, 9}
    половинки   m ∈ {1.5, 2.5, …, 9.5}
    випадковий  RANDOM_PER_DAY рівнів рівномірно в [лоу, хай] попереднього дня (контроль)
Рівні дня D+1 — усі круглі в [лоу D, хай D]. Подія (1m-свічки Binance, квіт–вер, кеш data/klines1m): рівень «заряджено»,
коли закриття відійшло на d = ATR(14) годинних барів, далі — перший дотик; результат — що раніше за 24 год: відскок на d
чи прохід на d. Дотик зверху = рівень як підтримка, знизу = як опір (для лонгу важливіший опір: де рух зупиняється).

КРИТЕРІЙ (до запуску): клас «реагує» з певного боку, якщо частка відскоків вища за випадкові рівні на ≥ 3 в. п., різниця
більша за 2 стандартні похибки і той самий знак в обох половинах періоду (до / після 01.07). Реагує хоч один клас →
наступний крок — ознаки («до найближчого круглого рівня вгорі, в ATR», «щойно пробито рівень») і перевірка моделі як
завжди. Жоден → ознаки не робимо.
→ reports/round_levels.md, reports/round_levels_events.csv

    python experiments/round_levels.py            # ~10–20 хв; --limit N — перші N монет
    python experiments/round_levels.py --d-atr 0.3   # дрібна реакція: відскок / прохід на 0.3 ATR → reports/round_levels_d0.3.md
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import klines1m as K
from profile_levels import atr1h, summary, table, touch

RANDOM_PER_DAY, SEED = 3, 42
SPLIT = pd.Timestamp("2026-07-01", tz="UTC")
CLASSES = {"великі": [1.0, 2.0, 5.0], "цілі": [3.0, 4.0, 6.0, 7.0, 8.0, 9.0], "половинки": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]}
OUT_MD = os.path.join("reports", "round_levels.md")
OUT_CSV = os.path.join("reports", "round_levels_events.csv")


def round_levels(lo: float, hi: float) -> list[tuple[str, float]]:
    """Усі круглі рівні всередині [lo, hi] з назвою класу."""
    out = []
    for k in range(int(np.floor(np.log10(lo))), int(np.floor(np.log10(hi))) + 1):
        for name, ms in CLASSES.items():
            for m in ms:
                L = m * 10.0 ** k
                if lo <= L <= hi:
                    out.append((name, L))
    return out


def day_events(base: str, m1: pd.DataFrame, rng, d_atr: float = 1.0) -> list[dict]:
    d1 = m1.resample("1D").agg({"h": "max", "l": "min"}).dropna()
    if len(d1) < 5:
        return []
    a1h = atr1h(m1)
    t = m1.index.as_unit("ns").asi8
    lo, hi, cl = m1["l"].to_numpy(), m1["h"].to_numpy(), m1["c"].to_numpy()
    out = []
    for D, nxt in zip(d1.index[:-1], d1.index[1:]):
        if nxt - D != pd.Timedelta(days=1):
            continue
        j = a1h.index.searchsorted(nxt, "right") - 1
        if j < 0 or not np.isfinite(a1h.iloc[j]):
            continue
        d = float(a1h.iloc[j]) * d_atr
        i0, i1 = np.searchsorted(t, nxt.value), np.searchsorted(t, (nxt + pd.Timedelta(days=1)).value)
        dlo, dhi = float(d1.at[D, "l"]), float(d1.at[D, "h"])
        if i1 - i0 < 600 or not (dlo > 0 and dhi > dlo):
            continue
        lv = round_levels(dlo, dhi) + [("випадковий", x) for x in rng.uniform(dlo, dhi, RANDOM_PER_DAY)]
        for kind, L in lv:
            r = touch(t, lo, hi, cl, L, d, i0, i1, len(t))
            if r:
                out.append({"base": base, "day": nxt, "level": kind, "L": L, "d": d, "d_pct": d / L, **r})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Круглі цінові рівні проти випадкових: відскок чи прохід після дотику (1m, 6 міс.)")
    ap.add_argument("--limit", type=int, default=0, help="лише перші N монет (перевірка)")
    ap.add_argument("--d-atr", type=float, default=1.0,
                    help="розмір реакції в частках ATR(14) 1h (за замовчуванням 1.0; 0.3 — дрібна реакція, звіт із суфіксом _d0.3)")
    args = ap.parse_args()
    sfx = "" if args.d_atr == 1.0 else f"_d{args.d_atr:g}"
    out_md, out_csv = OUT_MD.replace(".md", sfx + ".md"), OUT_CSV.replace(".csv", sfx + ".csv")
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    rows, bases = [], K.available()[: args.limit or None]
    for n, base in enumerate(bases, 1):
        m1 = K.load(base)
        if m1.empty:
            continue
        rows += day_events(base, m1, rng, args.d_atr)
        if n % 20 == 0:
            print(f"  {n}/{len(bases)} монет, подій {len(rows)}", flush=True)
    ev = pd.DataFrame(rows)
    ev["period"] = np.where(pd.to_datetime(ev["day"], utc=True) < SPLIT, "A", "B")
    ev.to_csv(out_csv, index=False)
    kinds = [*CLASSES, "випадковий"]

    def share(x: pd.DataFrame) -> tuple[float, float, int]:
        b, p = (x["result"] == "відскок").sum(), (x["result"] == "прохід").sum()
        if b + p == 0:
            return float("nan"), float("nan"), 0
        q = b / (b + p)
        return q, float(np.sqrt(q * (1 - q) / (b + p))), int(b + p)

    md = ["# Круглі цінові рівні проти випадкових", "",
          f"Монет: {ev['base'].nunique()}, 1m Binance, квітень–вересень 2026. Рівень «заряджено» відходом на d = {args.d_atr:g} × ATR(14) 1h; результат — "
          "відскок на d чи прохід на d за 24 год після першого дотику. Великі — 1 / 2 / 5 × 10^k, цілі — 3, 4, 6–9, половинки — x.5.", ""]
    verdict = []
    for side, role in (("знизу", "опір"), ("зверху", "підтримка")):
        x = ev[ev["side"] == side]
        md += [f"## Дотик {side} — рівень як {role}", "", table([{"група": lv, **summary(x[x["level"] == lv])} for lv in kinds]), ""]
        r0, s0, _ = share(x[x["level"] == "випадковий"])
        for lv in CLASSES:
            q, s, n = share(x[x["level"] == lv])
            diff, se = q - r0, float(np.sqrt(s ** 2 + s0 ** 2))
            per = []
            for p in "AB":
                qa, _, _ = share(x[(x["level"] == lv) & (x["period"] == p)])
                ra, _, _ = share(x[(x["level"] == "випадковий") & (x["period"] == p)])
                per.append(qa - ra)
            ok = diff >= 0.03 and diff > 2 * se and all(v > 0 for v in per)
            verdict.append(f"- {lv} · {role}: {q:.1%} проти {r0:.1%} у випадкових → {diff * 100:+.1f} в. п. (похибка ±{2 * se * 100:.1f}), "
                           f"періоди {per[0] * 100:+.1f} / {per[1] * 100:+.1f}, подій {n} → **{'РЕАГУЄ' if ok else 'ні'}**")
    md += ["## Критерій (≥ +3 в. п. проти випадкових, більше за похибку, обидва періоди)", ""] + verdict + [""]
    open(out_md, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
