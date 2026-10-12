"""
experiments/late_limit.py — режим split + лімітна заявка на відкаті в стані Late (2026-10-12, власник: «коли вже горить
Late, ставити лімітку нижче точки входу чи на ній»).

Late — почався епізод Alert основної моделі, а split не увійшов (модель «старт» на цьому барі не спрацювала; приклад —
CHIP 10-10). Варіант: на першому барі такого епізоду ставимо лімітну заявку L = закриття бару − x·ATR(15m), дійсну N барів.
Виконання — коли лоу наступних барів ≤ L (за ціною L або відкриттям, якщо бар відкрився нижче); стоп і ціль ½ — від ціни
виконання з ATR бару сигналу (як risk.LIMIT_ENTRY). Якщо раніше спрацює «старт» — звичайний вхід split, заявка знімається.
Заявка НЕ ставиться, коли Alert уже горів на попередньому барі (повторний сигнал після стопу) і коли угода відкрита.
Утримання і вихід — як у split (горить Alert основної або «старту»; ½ на +3 ATR → беззбиток; 4 бари без Alert; стоп k·ATR).

Критерій ДО запуску. Кандидати — два, зі слів власника: lim0 (x = 0, «на точці входу») і lim05 (x = 0.5, «нижче»),
обидва з N = 8 барів (2 год). Варіант проходить, якщо на тих самих 87 монетах ≥ 20 млн PF ≥ PF split і обидві половини
монет (md5) не гірші за split. lim1 (x = 1) і N = 4 — лише стійкість. Окремо дивимось ДОДАНІ угоди (ті, що увійшли
заявкою): їхній PF і медіана — це відповідь на «чи варто брати Late з відкату вручну».
Відомо раніше: ліміт −0.5 ATR у режимі market на 6 міс. — PF 1.02 проти 0.96, але в журналі гірше і пропускає половину
рухів ≥ +10% (long_backtest, pullback_grid); «пам'ять» старту повертала слабку половину монет (split_memory).

Дані — кеш experiments/random_entry.py (data/random_entry/tiers.pkl; спершу запустити його). Симулятор звіряється
з risk.position_state на звичайному split. Звіт — reports/late_limit.md.

    python experiments/late_limit.py
"""

import argparse
import hashlib
import os
import sys

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest  # noqa: E402
import strategies  # noqa: E402
import train  # noqa: E402
from random_entry import load_coins  # noqa: E402  (сусідній експеримент: кеш оцінок 87 монет)
from risk import PARTIAL_EXIT, atr, position_state  # noqa: E402
from tiers import split_scores  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "late_limit.md")
VARIANTS = {"split": None, "market": "market", "lim0": (0.0, 8), "lim05": (0.5, 8), "lim1": (1.0, 8),
            "lim0_4": (0.0, 4), "lim05_4": (0.5, 4)}


def pf(r) -> float:
    r = np.asarray(r, dtype=float)
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else float("nan")


def half(tk: str) -> str:
    return "A" if int(hashlib.md5(tk.encode()).hexdigest(), 16) % 2 else "B"


def simulate(op, price, high, low, a, main, ent, hold, k, exit_bars, part, lim) -> tuple[list[dict], int]:
    """Угоди по барах за правилами risk.position_state (вхід по ринку на барі, де ent ≥ 1) + лімітна заявка в стані Late.
    lim = (x, N) або None. Повертає (закриті угоди, скільки заявок поставлено)."""
    n, frac = len(price), part["frac"]
    out, orders = [], 0
    tr, order, below = None, None, 0
    for i in range(n):
        p = hold[i]
        if tr is None and order is not None:
            if i > order["until"]:
                order = None
            elif low[i] <= order["px"]:
                fill = min(op[i], order["px"]) if not np.isnan(op[i]) else order["px"]
                tr = {"i": i, "entry": fill, "stop": fill - k * order["atr"], "target": fill + part["atr"] * order["atr"],
                      "pp": None, "limit": True}
                order, below = None, 0
                continue
        if np.isnan(p) and tr is None:
            continue
        if tr is None:
            if not np.isnan(price[i]) and not np.isnan(a[i]):
                if not np.isnan(ent[i]) and ent[i] >= 1:                      # звичайний вхід режиму
                    tr = {"i": i, "entry": price[i], "stop": price[i] - k * a[i], "target": price[i] + part["atr"] * a[i],
                          "pp": None, "limit": False}
                    order, below = None, 0
                elif lim is not None and order is None and main[i] >= 1 and not (i > 0 and main[i - 1] >= 1):
                    order = {"px": price[i] - lim[0] * a[i], "atr": a[i], "until": i + lim[1]}      # старт епізоду Alert без входу = Late
                    orders += 1
            continue
        hit = low[i] <= tr["stop"]
        if not hit and tr["pp"] is None and high[i] >= tr["target"]:
            tr["pp"] = tr["target"]
            if part.get("breakeven"):
                tr["stop"] = max(tr["stop"], tr["entry"])
        if not np.isnan(p):
            below = below + 1 if p < 1 else 0
        if below >= exit_bars or hit:
            px = tr["stop"] if hit else price[i]
            rest = px / tr["entry"] - 1
            ret = (rest if tr["pp"] is None else frac * (tr["pp"] / tr["entry"] - 1) + (1 - frac) * rest) - backtest.FEE
            out.append({"i": tr["i"], "ret": ret, "stop": hit, "limit": tr["limit"]})
            tr, below = None, 0
    return out, orders


def main() -> None:
    argparse.ArgumentParser(description="split + лімітна заявка на відкаті в стані Late (Alert без входу моделі)").parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    data = load_coins(31)
    m15 = joblib.load(train.model_path(train.M15_NAME, "models"))
    k, exit_bars = m15["risk"]["k"], m15.get("exit_bars", 4)
    rule = strategies.entry_rule(strategies.get("market"))
    rows, placed, mismatch = [], {v: 0 for v in VARIANTS}, 0
    for base, (t, h) in data.items():
        idx = h.index
        op, price, high, low = (h[c].to_numpy(dtype=float) for c in ("open", "price", "high", "low"))
        a = atr(h).reindex(idx).to_numpy(dtype=float)
        main_s = t["alert_score"]
        entry_split, hold_split = split_scores(main_s, t["start_score"])
        main_a = np.nan_to_num(main_s.reindex(idx).to_numpy(dtype=float), nan=0.0)
        ent_a, hold_a = entry_split.reindex(idx).to_numpy(dtype=float), hold_split.reindex(idx).to_numpy(dtype=float)
        hold_m = main_s.reindex(idx).to_numpy(dtype=float)
        for name, lim in VARIANTS.items():
            if lim == "market":
                trades, n_ord = simulate(op, price, high, low, a, main_a, hold_m, hold_m, k, exit_bars, PARTIAL_EXIT, None)
            else:
                trades, n_ord = simulate(op, price, high, low, a, main_a, ent_a, hold_a, k, exit_bars, PARTIAL_EXIT, lim)
            placed[name] += n_ord
            for x in trades:
                rows.append({"v": name, "ticker": base, "ts": idx[x["i"]], "ret": x["ret"], "stop": x["stop"], "limit": x["limit"]})
            if name == "split":                                               # звірка з робочим рушієм угод
                pos = position_state(h, hold_split, 1.0, k, exit_bars, PARTIAL_EXIT, rule, None, None, entry_prob=entry_split)
                ref = [(x.get("result_pct") or 0) - backtest.FEE for x in pos.get("trades", [])]
                if len(ref) != len(trades) or (ref and not np.allclose(ref, [x["ret"] for x in trades], atol=1e-9)):
                    mismatch += 1
    d = pd.DataFrame(rows)
    d["half"] = d["ticker"].map(half)
    mid = d["ts"].min() + (d["ts"].max() - d["ts"].min()) / 2
    d["per"] = np.where(d["ts"] < mid, 1, 2)
    lines = ["# split + лімітна заявка на відкаті в стані Late (12.10)", "",
             f"{len(data)} монет ≥ 20 млн поза навчанням, вікно контрольного бектесту (з 11.09), вихід як у робочому режимі, комісія 0.1%.",
             "limX — на першому барі епізоду Alert без входу split: заявка на закриття бару − X·ATR, дійсна 8 барів (варіанти _4 — 4 бари).", "",
             "| Варіант | Угод | PF | PF без топ-3 | У плюсі | По стопу | Сума | Монети A / B | Час 1-а / 2-а | Заявок / виконано | ДОДАНІ угоди: PF · медіана · по стопу |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for name in VARIANTS:
        g = d[d.v == name]
        r = g["ret"].to_numpy()
        add = g[g.limit]
        res[name] = {"pf": pf(r), "A": pf(g[g.half == "A"].ret), "B": pf(g[g.half == "B"].ret)}
        extra = (f"{placed[name]} / {len(add)} | {pf(add.ret):.2f} · {add.ret.median():+.2%} · {add.stop.mean():.0%}"
                 if len(add) else "— | —")
        lines.append(f"| {name} | {len(g)} | {pf(r):.2f} | {pf(np.sort(r)[:-3]):.2f} | {(r > 0).mean():.0%} | {g.stop.mean():.0%} | "
                     f"{r.sum():+.0%} | {res[name]['A']:.2f} / {res[name]['B']:.2f} | "
                     f"{pf(g[g.per == 1].ret):.2f} / {pf(g[g.per == 2].ret):.2f} | {extra} |")
    s = res["split"]
    lines.append("")
    for name in ("lim0", "lim05"):
        m = res[name]
        ok = m["pf"] >= s["pf"] and m["A"] >= s["A"] and m["B"] >= s["B"]
        lines.append(f"**Критерій {name} (PF ≥ split і обидві половини монет не гірші): {'ВИКОНАНО' if ok else 'НЕ ВИКОНАНО'}** "
                     f"(PF {m['pf']:.2f} проти {s['pf']:.2f}; A {m['A']:.2f} / {s['A']:.2f}; B {m['B']:.2f} / {s['B']:.2f})")
    lines += ["", f"Звірка симулятора з risk.position_state на звичайному split: розбіжностей {mismatch} з {len(data)}."]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
