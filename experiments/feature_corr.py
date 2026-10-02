"""
experiments/feature_corr.py — як 15m-ознаки моделі корелюють між собою (2026-10-02).

Кореляція Спірмена (рангова: стійка до викидів, ловить будь-яку монотонну залежність) між 34 ознаками FEATURES_15M
на 15m-барах усіх монет з кешу experiments/outcome_target.py (тікери навчання + монети бектесту).
Навіщо: ознаки, що майже дублюють одна одну (|ρ| ≥ 0.8), ділять між собою «важливість» у LightGBM (звіт важливості вводить
в оману), а в логрегресії розхитують коефіцієнти; кандидати на спрощення моделі.

Вихід: найсильніші пари в консолі + reports/feature_corr.html — теплова карта (ознаки впорядковані кластеризацією,
схожі — поруч; наведення — значення).

    python experiments/feature_corr.py
"""

import argparse
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from features import FEATURES, FEATURES_15M

CACHE = os.path.join("reports", "backtests", "_outcome_feats.pkl")
OUT = os.path.join("reports", "feature_corr.html")


def color(v: float) -> str:
    """Розбіжна шкала: синій (−1) — сірий (0) — помаранчевий (+1)."""
    t = min(1.0, abs(v))
    a = (60, 120, 216) if v < 0 else (216, 90, 48)
    g = (60, 60, 58)
    return "rgb({},{},{})".format(*(round(g[i] + (a[i] - g[i]) * t) for i in range(3)))


def heatmap(c: pd.DataFrame, n_rows: int) -> str:
    names = list(c.columns)
    n, cell, left, top = len(names), 18, 150, 150
    w, h = left + n * cell + 20, top + n * cell + 60
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="IBM Plex Sans, system-ui, sans-serif" font-size="11">']
    for i, name in enumerate(names):
        y = top + i * cell + cell * 0.7
        mark = "#f0997b" if name not in FEATURES else "#a8a79f"
        parts.append(f'<text x="{left - 6}" y="{y}" text-anchor="end" fill="{mark}">{html.escape(name)}</text>')
        x = left + i * cell + cell * 0.7
        parts.append(f'<text x="{x}" y="{top - 6}" fill="{mark}" transform="rotate(-60 {x} {top - 6})">{html.escape(name)}</text>')
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            v = c.iat[i, j]
            parts.append(f'<rect x="{left + j * cell}" y="{top + i * cell}" width="{cell - 1}" height="{cell - 1}" rx="2" '
                         f'fill="{color(v)}"><title>{html.escape(a)} × {html.escape(b)}: ρ = {v:+.2f}</title></rect>')
            if i != j and abs(v) >= 0.8:
                parts.append(f'<text x="{left + j * cell + cell / 2 - 0.5}" y="{top + i * cell + cell * 0.68}" text-anchor="middle" '
                             f'fill="#fff" font-size="8">{abs(v):.1f}</text>')
    y0 = top + n * cell + 22
    for k, v in enumerate(np.linspace(-1, 1, 21)):
        parts.append(f'<rect x="{left + k * 14}" y="{y0}" width="13" height="10" fill="{color(v)}"/>')
    parts.append(f'<text x="{left}" y="{y0 + 24}" fill="#a8a79f">−1</text><text x="{left + 140}" y="{y0 + 24}" fill="#a8a79f">0</text>'
                 f'<text x="{left + 284}" y="{y0 + 24}" fill="#a8a79f">+1</text>')
    parts.append("</svg>")
    return f"""<!doctype html><html lang="uk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Кореляція ознак</title><style>
body {{ margin: 0; padding: 20px 16px; background: #0f0f0e; color: #f0efec; font: 13px/1.5 "IBM Plex Sans", system-ui, sans-serif; }}
h1 {{ font-size: 16px; margin: 0 0 6px; }} p {{ color: #a8a79f; max-width: 820px; margin: 0 0 14px; }}
.wrap {{ overflow-x: auto; }}
</style></head><body>
<h1>Кореляція 15m-ознак моделі (Спірмен)</h1>
<p>{n_rows:,} барів 15m, усі монети з даними. Помаранчеве — ознаки рухаються разом, синє — протилежно, сіре — незалежні.
Схожі ознаки стоять поруч (кластеризація). Цифри в клітинках — пари з |ρ| ≥ 0.8 (майже дублі). Назви помаранчевим — короткі
15m-ознаки, сірим — годинні (23 базові). Наведи на клітинку — точне значення.</p>
<div class="wrap">{"".join(parts)}</div></body></html>"""


def main() -> None:
    argparse.ArgumentParser(description="Кореляція 15m-ознак між собою: найсильніші пари і теплова карта").parse_args()
    d = pd.read_pickle(CACHE)[FEATURES_15M]
    d = d.sample(min(len(d), 300_000), random_state=0)
    c = d.corr(method="spearman")
    # порядок — ієрархічна кластеризація за 1 − |ρ|: схожі ознаки поруч
    dist = squareform((1 - c.abs()).clip(lower=0).to_numpy(), checks=False)
    order = [c.columns[i] for i in leaves_list(linkage(dist, "average"))]
    c = c.loc[order, order]
    os.makedirs("reports", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(heatmap(c, len(d)))
    pairs = c.where(np.triu(np.ones(c.shape, dtype=bool), 1)).stack().rename("rho").reset_index()
    pairs = pairs.reindex(pairs["rho"].abs().sort_values(ascending=False).index)
    print(f"{len(d):,} барів 15m; пар з |ρ| ≥ 0.8: {(pairs['rho'].abs() >= 0.8).sum()}, ≥ 0.6: {(pairs['rho'].abs() >= 0.6).sum()} "
          f"з {len(pairs)}\n\nНайсильніші пари:")
    for r in pairs.head(25).itertuples():
        print(f"  {r.level_0:<18} {r.level_1:<18} {r.rho:+.2f}")
    mean_abs = (c.abs().sum() - 1) / (len(c) - 1)
    print("\nНайнезалежніші ознаки (середня |ρ| з рештою):")
    for f, v in mean_abs.sort_values().head(8).items():
        print(f"  {f:<18} {v:.2f}")
    print(f"\nТеплова карта: {OUT}")


if __name__ == "__main__":
    main()
