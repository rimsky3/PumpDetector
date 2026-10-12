"""experiments/entry_kind.py — розвідка (без критерію): угоди split на 87 монетах контрольного бектесту — за типом входу: на хаях чи з відкату в гарячій монеті.
"""

import argparse
import os
import sys

argparse.ArgumentParser(description="Розвідка: угоди split за типом входу (на хаях / з відкату) і сетапи в розмітці").parse_args()

ROOT = r"D:\PD tracker"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import joblib
import numpy as np
import pandas as pd

import train
from late_limit import pf, simulate
from random_entry import load_coins
from risk import PARTIAL_EXIT, atr
from tiers import split_scores

# розмітка: скільки інтервалів кожного сетапу
p = pd.read_csv("pumps.csv")
pp = p[p.pattern_type == "pump"]
print("розмітка, імпульси за сетапом:", pp["setup"].fillna("(не вказано)").value_counts().to_dict() if "setup" in pp else "колонки setup немає")
print("Failed impulse:", int((p.pattern_type == "false_start").sum()), "| з полем «мій вхід»:", int(p["entry_ts"].notna().sum()) if "entry_ts" in p else 0)
# друга хвиля = імпульс, що почався ≤ 24 год після кінця попереднього імпульсу того ж тікера
pp = pp.assign(s=pd.to_datetime(pp.start_ts), e=pd.to_datetime(pp.end_ts)).sort_values(["base_asset", "s"])
gap = (pp["s"] - pp.groupby("base_asset")["e"].shift(1)).dt.total_seconds() / 3600
print(f"імпульсів, що почались ≤ 24 год після кінця попереднього на тому ж тікері: {(gap <= 24).sum()} з {len(pp)}; ≤ 6 год: {(gap <= 6).sum()}")

data = load_coins(31)
m15 = joblib.load(train.model_path(train.M15_NAME, "models"))
k, exit_bars = m15["risk"]["k"], m15.get("exit_bars", 4)
rows = []
for base, (t, h) in data.items():
    idx = h.index
    op, price, high, low = (h[c].to_numpy(dtype=float) for c in ("open", "price", "high", "low"))
    a = atr(h).reindex(idx).to_numpy(dtype=float)
    main_s = t["alert_score"]
    ent, hold = split_scores(main_s, t["start_score"])
    main_a = np.nan_to_num(main_s.reindex(idx).to_numpy(dtype=float), nan=0.0)
    trades, _ = simulate(op, price, high, low, a, main_a, ent.reindex(idx).to_numpy(dtype=float), hold.reindex(idx).to_numpy(dtype=float),
                         k, exit_bars, PARTIAL_EXIT, None)
    hi24 = pd.Series(high).rolling(96, min_periods=48).max().to_numpy()
    lo24 = pd.Series(low).rolling(96, min_periods=48).min().to_numpy()
    for x in trades:
        i = x["i"]
        if i < 96 or np.isnan(hi24[i]):
            continue
        rows.append({"ret": x["ret"], "stop": x["stop"], "from_hi": price[i] / hi24[i] - 1, "ret24": price[i] / price[i - 96] - 1,
                     "from_lo": price[i] / lo24[i] - 1, "run4h": price[i] / price[i - 16] - 1})
d = pd.DataFrame(rows)
print(f"\nугод split з історією 24 год: {len(d)}, PF {pf(d.ret):.2f}")


def show(name, m):
    g = d[m]
    print(f"  {name:<58} угод {len(g):>4} | PF {pf(g.ret):.2f} | медіана {g.ret.median():+.2%} | у плюсі {(g.ret > 0).mean():.0%} | по стопу {g.stop.mean():.0%}")


print("за відстанню від хаю 24 год на вході:")
show("на хаях (≤ 1% від хаю 24 год)", d.from_hi > -0.01)
show("трохи нижче (1–3%)", (d.from_hi <= -0.01) & (d.from_hi > -0.03))
show("з відкату (3–8% нижче хаю)", (d.from_hi <= -0.03) & (d.from_hi > -0.08))
show("глибокий відкат (> 8% нижче хаю)", d.from_hi <= -0.08)
print("гаряча монета (вже ≥ +10% від лоу 24 год) і тип входу:")
hot = d.from_lo >= 0.10
show("гаряча + на хаях (≤ 1%)", hot & (d.from_hi > -0.01))
show("гаряча + з відкату (≥ 3% нижче хаю) — як RSR 2-й вхід", hot & (d.from_hi <= -0.03))
show("не гаряча (< +10% від лоу)", ~hot)
print("за ростом за останні 4 год перед входом:")
show("вже +8% і більше за 4 год (вхід на імпульсі)", d.run4h >= 0.08)
show("+3…+8% за 4 год", (d.run4h >= 0.03) & (d.run4h < 0.08))
show("менше +3% за 4 год (вхід до імпульсу)", d.run4h < 0.03)
