"""
experiments/exit_examples.py — наочно: де вийшло б поточне правило («Alert згас 4 бари») і нове
(«Alert згас 4 бари І (ціна віддала ½ руху від максимуму з входу АБО рух < 2 ATR)») на тих самих угодах.

Входи — out-of-fold Alert (як у exit_giveback.py). Беремо угоди, де виходи різняться: найбільший виграш нового
правила, найбільший програш і кілька між ними. Сторінка — reports/exit_rule_examples.html (SVG, без бібліотек).

    python experiments/exit_examples.py
"""

import argparse
import html
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
from experiments.exit_giveback import simulate

B = pd.Timedelta(minutes=15)
W, HT, PAD = 620, 280, 46


def panel(h, sc, cur, new, k, title):
    t0 = min(cur["t0"], new["t0"]) - pd.Timedelta(hours=3)
    t1 = max(cur["t1"], new["t1"]) + pd.Timedelta(hours=3)
    v = h[(h.index >= t0) & (h.index <= t1)]
    s = sc.reindex(v.index)
    lo, hi = v["low"].min(), v["high"].max()
    pk = v.loc[(v.index >= new["t0"]) & (v.index <= new["t1"]), "high"].max()
    stop = new["e"] - k * new["atr"]
    lo = min(lo, stop)
    n = len(v)
    plot_h = HT - 70
    x = lambda i: PAD + (i + 0.5) * (W - PAD - 10) / n                          # noqa: E731
    y = lambda p: 22 + (hi - p) / (hi - lo) * plot_h if hi > lo else 22 + plot_h / 2  # noqa: E731
    bw = max(1.5, (W - PAD - 10) / n * 0.6)
    out = [f'<svg viewBox="0 0 {W} {HT}" width="{W}" height="{HT}" role="img" aria-label="{html.escape(title)}">',
           f'<text x="8" y="15" class="t">{html.escape(title)}</text>']
    # смужка рівня сигналу внизу: червоне — Alert, жовте — Кандидат
    for i, (ts, val) in enumerate(s.items()):
        if pd.notna(val) and val >= 1:
            out.append(f'<rect x="{x(i) - bw / 2:.1f}" y="{HT - 40}" width="{bw:.1f}" height="6" class="al"/>')
    for i, r in enumerate(v.itertuples()):
        up = r.price >= r.open
        out.append(f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(r.high):.1f}" y2="{y(r.low):.1f}" class="{"u" if up else "d"}"/>')
        y0, y1 = sorted([y(r.open), y(r.price)])
        out.append(f'<rect x="{x(i) - bw / 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(0.8, y1 - y0):.1f}" class="{"u" if up else "d"}"/>')
    def idx(ts):
        return int(np.clip(v.index.get_indexer([ts], method="nearest")[0], 0, n - 1))
    ie, ic, inew = idx(new["t0"]), idx(cur["t1"]), idx(new["t1"])
    out.append(f'<line x1="{x(ie):.1f}" x2="{W - 10}" y1="{y(stop):.1f}" y2="{y(stop):.1f}" class="stop"/>'
               f'<text x="{W - 12}" y="{y(stop) - 3:.1f}" class="lab r">стоп</text>')
    half = pk - 0.5 * (pk - new["e"])
    if pk > new["e"]:
        out.append(f'<line x1="{x(ie):.1f}" x2="{x(inew):.1f}" y1="{y(half):.1f}" y2="{y(half):.1f}" class="half"/>'
                   f'<text x="{x(ie) + 4:.1f}" y="{y(half) - 3:.1f}" class="lab b">½ руху від максимуму</text>')
    out.append(f'<path d="M{x(ie):.1f},{y(v.iloc[ie]["low"]) + 14:.1f} l-6,9 h12 z" class="entry"/>'
               f'<text x="{x(ie):.1f}" y="{y(v.iloc[ie]["low"]) + 34:.1f}" class="lab c">вхід</text>')
    for i, cls, lab in ((ic, "cur", f'зараз {cur["ret"]:+.1%}'), (inew, "new", f'нове {new["ret"]:+.1%}')):
        yy = y(v.iloc[i]["high"]) - 10
        out.append(f'<path d="M{x(i) - 6:.1f},{yy - 9:.1f} h12 l-6,9 z" class="{cls}"/>'
                   f'<text x="{x(i):.1f}" y="{yy - 13:.1f}" class="lab c {cls}t">{lab}</text>')
    out.append(f'<text x="{PAD}" y="{HT - 22}" class="lab">червона смужка — Alert на барі</text>')
    out.append(f'<text x="{PAD - 4}" y="{y(hi) + 4:.1f}" class="lab r">{hi:.4g}</text><text x="{PAD - 4}" y="{y(lo):.1f}" class="lab r">{lo:.4g}</text>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    argparse.ArgumentParser(description="Приклади угод: поточний вихід проти нового (SVG-сторінка)").parse_args()
    load_dotenv(".env")
    oof = pd.read_parquet(os.path.join(train.MODELS_DIR, "oof_15m.parquet"))
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    b15 = joblib.load(train.model_path(train.M15_NAME))
    k, share = b15["risk"]["k"], b15.get("confirm", {}).get("share", 0.75)
    oof["score"] = np.minimum(oof["p15"] / oof["thr15"], oof["s1h"].fillna(0) / share)
    pairs = []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for b in sorted(oof["base_asset"].unique()):
            h = aggregate(load_raw(conn, [b], INTERVAL_15M), b, "15min").dropna(subset=["price", "high", "low"])
            sc = oof[oof["base_asset"] == b].set_index("ts")["score"]
            cur = {t["t0"]: t for t in simulate(h, sc, k, "alert4", 0.5, 2, False)}
            new = simulate(h, sc, k, "and_small", 0.5, 2, False)
            from risk import atr
            a = atr(h)
            for t in new:
                c = cur.get(t["t0"])
                if c is None or c["t1"] == t["t1"]:
                    continue
                t["e"] = h.at[t["t0"], "price"]; t["atr"] = a.get(t["t0"])
                pairs.append((b, h, sc, c, t))
    pairs.sort(key=lambda p: p[4]["ret"] - p[3]["ret"])
    worse, better = pairs[:3], pairs[-4:][::-1]
    n_better = sum(1 for p in pairs if p[4]["ret"] > p[3]["ret"])
    secs = []
    for name, group in (("Де нове правило краще", better), ("Де нове правило гірше", worse)):
        cards = "".join(panel(h, sc, c, t, k, f"{b} · вхід {t['t0']:%m-%d %H:%M} · зараз {c['ret']:+.1%} → нове {t['ret']:+.1%}")
                        for b, h, sc, c, t in group)
        secs.append(f"<h2>{name}</h2><div class='grid'>{cards}</div>")
    page = f"""<!doctype html><html lang="uk"><head><meta charset="utf-8"><title>Вихід: зараз проти нового</title>
<style>body{{background:#121418;color:#f2f4f7;font:14px system-ui,sans-serif;margin:24px}}h1{{font-size:20px}}h2{{font-size:16px;margin-top:28px}}
p{{color:#b3bac6;max-width:900px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax({W}px,1fr));gap:14px}}
svg{{background:#1a1d22;border:1px solid #2c3139;border-radius:8px}}.t{{fill:#f2f4f7;font-size:12px;font-weight:600}}
.u{{fill:#3fb56f;stroke:#3fb56f}}.d{{fill:#e66767;stroke:#e66767}}.al{{fill:#e66767;opacity:.8}}
.stop{{stroke:#e0a526;stroke-dasharray:4 3}}.half{{stroke:#3987e5;stroke-dasharray:2 3}}.entry{{fill:#3fb56f}}
.cur{{fill:#b3bac6}}.new{{fill:#3987e5}}.lab{{fill:#7e8794;font-size:10px}}.lab.r{{text-anchor:end}}.lab.c{{text-anchor:middle}}
.lab.b{{fill:#3987e5}}.curt{{fill:#b3bac6;font-size:11px}}.newt{{fill:#3987e5;font-size:11px}}</style></head><body>
<h1>Вихід з позиції: зараз (сірий ▼) проти нового (синій ▼)</h1>
<p>Зараз: вихід, коли Alert згас 4 бари 15m поспіль. Нове: Alert згас 4 бари <b>і</b> ціна віддала ½ руху від максимуму з моменту
входу (синій пунктир — рівень ½ на момент виходу) — або рух від входу ще менший за 2 ATR. Стоп (жовтий пунктир) однаковий.
Входи — out-of-fold сигнали моделі, комісія 0.1%. Угод, де виходи різняться: {len(pairs)}, нове краще в {n_better}.</p>
{''.join(secs)}</body></html>"""
    path = os.path.join("reports", "exit_rule_examples.html")
    open(path, "w", encoding="utf-8").write(page)
    print(f"{path}: {len(pairs)} угод з різним виходом, нове краще в {n_better}")


if __name__ == "__main__":
    main()
