"""
experiments/split_memory.py — режим split з «пам'яттю» старту (2026-10-12, власник: CHIP 10-10 — «чому немає входу?»).

Проблема: у split вхід лише на барі, де ОДНОЧАСНО горять модель «старт» і годинне підтвердження. Годинна запізнюється на
30–45 хв, а «старт» за цей час гасне (CHIP: «старт» 16:15 = 0.967 при годинній 0.01; годинна 16:45 = 1.06 при «старті» 0.26)
→ рух пропущено. Варіант memN: вхід, коли горить Alert основної моделі (15m + годинна), а «старт» БЕЗ годинного фільтра
був ≥ свого порогу на будь-якому з останніх N барів 15m (включно з поточним). Утримання і вихід — як у split.

Критерій ДО запуску (основний варіант mem4 = 1 год): монети ≥ 20 млн, ті самі 87 монет і дані, що контрольний бектест
12.10 (reports/backtests/20261011_2228_current_split.json); PF mem4 ≥ PF split і обидві половини монет (md5) не гірші.
mem2 / mem8 — лише стійкість. Звіт — reports/split_memory.md.

    python experiments/split_memory.py                 # ~3–5 хв, лише база (без завантажень)
"""

import argparse
import hashlib
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest  # noqa: E402
import strategies  # noqa: E402
import train  # noqa: E402
from risk import PARTIAL_EXIT, position_state  # noqa: E402
from tiers import split_scores  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "reports", "backtests", "20261011_2228_current_split.json")
OUT = os.path.join(ROOT, "reports", "split_memory.md")


def pf(r: pd.Series) -> float:
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else float("nan")


def half(tk: str) -> str:
    return "A" if int(hashlib.md5(tk.encode()).hexdigest(), 16) % 2 else "B"


def variants(t: pd.DataFrame) -> dict[str, tuple[pd.Series | None, pd.Series]]:
    """{назва: (вхід, утримання)} — вхід None = вхід за утриманням (market)."""
    main = t["alert_score"]
    entry_split, hold = split_scores(main, t["start_score"])
    out = {"market": (None, main), "split": (entry_split, hold)}
    for n in (2, 4, 8):
        fired = (t["start_s15"] >= 1).astype(float).rolling(n, min_periods=1).max() >= 1
        out[f"mem{n}"] = (main.where(fired, 0.0), hold)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="split з «пам'яттю» старту: вхід на Alert основної, якщо «старт» спрацював за N барів")
    p.add_argument("--days", type=int, default=30, help="вікно, днів (як у контрольному бектесті)")
    args = p.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    coins = [x["ticker"] for x in json.load(open(REF, encoding="utf-8"))["per_ticker"]]
    m15 = joblib.load(train.model_path(train.M15_NAME, "models"))
    m15["start"] = joblib.load(train.model_path(train.START_NAME, "models"))
    m1 = joblib.load(train.model_path("lightgbm", "models"))
    k, rule = m15.get("risk", {}).get("k"), strategies.entry_rule(strategies.get("market"))
    rows = []
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for i, base in enumerate(coins, 1):
            th = backtest.tiers_and_bars(conn, base, m15, m1, args.days)
            if th is None:
                continue
            t, h = th
            for name, (entry_p, score) in variants(t).items():
                pos = position_state(h, score, 1.0, k, m15.get("exit_bars", 4), PARTIAL_EXIT, rule, None, None, entry_prob=entry_p)
                for tr in pos.get("trades", []):
                    rows.append({"v": name, "ticker": base, "entry_ts": tr["entry_ts"], "reason": tr.get("reason"),
                                 "ret": (tr.get("result_pct") or 0) - backtest.FEE})
            if i % 10 == 0:
                print(f"  {i}/{len(coins)}", flush=True)
    d = pd.DataFrame(rows)
    d["half"] = d["ticker"].map(half)
    ts = pd.to_datetime(d["entry_ts"], utc=True)
    mid = ts.min() + (ts.max() - ts.min()) / 2
    d["per"] = np.where(ts < mid, "1-а", "2-а")
    lines = ["# split з «пам'яттю» старту (12.10)", "",
             f"{len(coins)} монет ≥ 20 млн поза навчанням (як контрольний бектест 12.10), {args.days} днів, вихід як у робочому режимі, комісія 0.1%.",
             "memN — вхід на Alert основної моделі, якщо «старт» (без годинного фільтра) був ≥ порогу за останні N барів 15m.", "",
             "| Варіант | Угод | PF | PF без топ-3 | У плюсі | По стопу | Сума | Монети A / B | Час 1-а / 2-а |", "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for name in ["market", "split", "mem2", "mem4", "mem8"]:
        g = d[d["v"] == name]
        r = g["ret"]
        res[name] = {"pf": pf(r), "A": pf(g[g.half == "A"].ret), "B": pf(g[g.half == "B"].ret)}
        lines.append(f"| {name} | {len(g)} | {pf(r):.2f} | {pf(r.sort_values().iloc[:-3]):.2f} | {(r > 0).mean():.0%} | "
                     f"{(g.reason == 'stop').mean():.0%} | {r.sum():+.0%} | {res[name]['A']:.2f} / {res[name]['B']:.2f} | "
                     f"{pf(g[g.per == '1-а'].ret):.2f} / {pf(g[g.per == '2-а'].ret):.2f} |")
    s, m = res["split"], res["mem4"]
    ok = m["pf"] >= s["pf"] and m["A"] >= s["A"] and m["B"] >= s["B"]
    lines += ["", f"**Критерій (mem4 ≥ split за PF і в обох половинах монет): {'ВИКОНАНО' if ok else 'НЕ ВИКОНАНО'}** "
              f"(PF {m['pf']:.2f} проти {s['pf']:.2f}; A {m['A']:.2f} / {s['A']:.2f}; B {m['B']:.2f} / {s['B']:.2f})"]
    # угоди, які дає лише пам'ять (у split їх немає)
    key = ["ticker", "entry_ts"]
    extra = d[d.v == "mem4"].merge(d[d.v == "split"][key], on=key, how="left", indicator=True).query("_merge == 'left_only'")
    if len(extra):
        lines.append(f"\nЛише mem4 (немає в split): {len(extra)} угод, PF {pf(extra.ret):.2f}, сума {extra.ret.sum():+.0%}, "
                     f"медіана {extra.ret.median():+.2%}.")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
