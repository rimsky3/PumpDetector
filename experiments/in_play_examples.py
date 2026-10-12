"""
experiments/in_play_examples.py — ПРОФІЛЬ ПРИКЛАДІВ «В ГРІ» від власника проти типового входу моделі (2026-10-12).
Власник: «я дам тобі приклад, а ти визнач параметри» — MAGIC 09.10 і OGN 08.10. Розвідка, БЕЗ критерію: міряємо приклади
на різних годинах від старту руху і дивимось, за якими параметрами вони виділяються серед звичайних входів моделі
(угоди split контрольного бектесту, 87 монет, кеш experiments/random_entry.py). Параметри, де обидва приклади стабільно у
верхніх 10% — кандидати у визначення «в грі»; перевірка визначення — окремо, на 6 міс. з критерієм.

Параметри на момент t (усе з 15m-барів до t; «звичайний оборот» — медіана денного обороту за до 30 повних днів до t):
    usual_turn   звичайний денний оборот, млн $ (яка це монета: тиха чи ліквідна)
    vol24_x      оборот за 24 год ÷ звичайний
    vol4_x       оборот за 4 год × 6 ÷ звичайний (темп зараз)
    hot_hours    скільки з останніх 24 год погодинний оборот був ≥ 3 × звичайного погодинного (як довго монета «жива»)
    oi_4h, oi_24h   зміна OI ($) за 4 і 24 год
    oi_hold      найбільше просідання OI від його максимуму за 12 год (0 = OI тільки росте)
    ret_4h, ret_24h зміна ціни
    from_hi      ціна відносно хаю 24 год
    pull_12h     найглибший відкат від поточного максимуму за 12 год, у частках руху від лоу 24 год (сходинки чи пила)
    oi_turn      OI ÷ оборот за 24 год

    python experiments/in_play_examples.py                       # MAGIC і OGN (локальна база)
    python experiments/in_play_examples.py --add "RSR 2026-10-12 04:30"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from features import aggregate, load_raw
from late_limit import simulate
from random_entry import load_coins
from risk import PARTIAL_EXIT, atr
from tiers import split_scores

EXAMPLES = [("MAGIC", "2026-10-09 09:00"), ("OGN", "2026-10-08 08:15")]
HOURS = [0, 2, 4, 8, 12, 24]
COLS = ["usual_turn", "vol24_x", "vol4_x", "hot_hours", "oi_4h", "oi_24h", "oi_hold", "ret_4h", "ret_24h", "from_hi", "pull_12h", "oi_turn"]
FMT = {"usual_turn": "{:.1f}", "vol24_x": "×{:.1f}", "vol4_x": "×{:.1f}", "hot_hours": "{:.0f}", "oi_4h": "{:+.0%}", "oi_24h": "{:+.0%}",
       "oi_hold": "{:+.0%}", "ret_4h": "{:+.0%}", "ret_24h": "{:+.0%}", "from_hi": "{:+.1%}", "pull_12h": "{:.2f}", "oi_turn": "{:.2f}"}


def params(h: pd.DataFrame) -> pd.DataFrame:
    """Параметри на закритті кожного 15m-бару (індекс — відкриття бару)."""
    v, p, oi = h["volume"], h["price"], h["oi"]
    day = v.resample("1D").sum()
    usual_d = day.rolling(30, min_periods=7).median().shift(1)
    usual = pd.Series(usual_d.reindex(h.index.floor("1D")).to_numpy(), index=h.index)
    out = pd.DataFrame(index=h.index)
    out["usual_turn"] = usual / 1e6
    t24 = v.rolling(96, min_periods=48).sum()
    out["vol24_x"] = t24 / usual
    out["vol4_x"] = v.rolling(16, min_periods=8).sum() * 6 / usual
    hourly = v.rolling(4, min_periods=4).sum()
    out["hot_hours"] = (hourly >= 3 * usual / 24).rolling(96, min_periods=24).sum() / 4
    out["oi_4h"] = oi / oi.shift(16) - 1
    out["oi_24h"] = oi / oi.shift(96) - 1
    out["oi_hold"] = oi / oi.rolling(48, min_periods=8).max() - 1
    out["ret_4h"] = p / p.shift(16) - 1
    out["ret_24h"] = p / p.shift(96) - 1
    hi24, lo24 = h["high"].rolling(96, min_periods=24).max(), h["low"].rolling(96, min_periods=24).min()
    out["from_hi"] = p / hi24 - 1
    run_hi = h["high"].rolling(48, min_periods=8).max()
    dd = (run_hi - h["low"]).rolling(48, min_periods=8).max()          # наближення: найбільший розрив між хаєм 12 год і лоу бару
    out["pull_12h"] = dd / (hi24 - lo24).where(hi24 > lo24)
    out["oi_turn"] = oi / t24
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Профіль прикладів «в грі» (MAGIC, OGN) проти типового входу моделі")
    ap.add_argument("--add", nargs="*", default=[], help='ще приклади: "ТІКЕР YYYY-MM-DD HH:MM" (старт руху, UTC)')
    args = ap.parse_args()
    ex = EXAMPLES + [(a.split()[0].upper(), " ".join(a.split()[1:])) for a in args.add]
    load_dotenv(".env")
    # типовий вхід моделі: угоди split контрольного бектесту
    m15 = joblib.load(train.model_path(train.M15_NAME, "models"))
    k, exit_bars = m15["risk"]["k"], m15.get("exit_bars", 4)
    ref = []
    for base, (t, h) in load_coins(31).items():
        idx = h.index
        op, price, high, low = (h[c].to_numpy(dtype=float) for c in ("open", "price", "high", "low"))
        a = atr(h).reindex(idx).to_numpy(dtype=float)
        ent, hold = split_scores(t["alert_score"], t["start_score"])
        main_a = np.nan_to_num(t["alert_score"].reindex(idx).to_numpy(dtype=float), nan=0.0)
        trades, _ = simulate(op, price, high, low, a, main_a, ent.reindex(idx).to_numpy(dtype=float), hold.reindex(idx).to_numpy(dtype=float),
                             k, exit_bars, PARTIAL_EXIT, None)
        if trades:
            pr = params(h)
            for x in trades:
                ref.append({**pr.iloc[x["i"]].to_dict(), "ret": x["ret"]})
    ref = pd.DataFrame(ref).dropna(subset=["usual_turn"])
    print(f"типовий вхід моделі: {len(ref)} угод split, 87 монет ≥ 20 млн\n")
    rows = []
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=10) as conn:
        for tk, start in ex:
            t0 = pd.Timestamp(start, tz="UTC")
            h = aggregate(load_raw(conn, [tk], "15min", since=t0 - pd.Timedelta(days=40)), tk, "15min")
            if h.empty or h.index.max() < t0:
                print(f"! {tk}: немає 15m-даних у локальній базі на {start}")
                continue
            pr = params(h)
            for hrs in HOURS:
                ts = t0 + pd.Timedelta(hours=hrs)
                if ts in pr.index:
                    rows.append({"приклад": f"{tk} +{hrs} год", **pr.loc[ts].to_dict(), "ціна": h.at[ts, "price"]})
    e = pd.DataFrame(rows).set_index("приклад")
    q = ref[COLS].quantile([0.1, 0.5, 0.9])
    head = "| Параметр | " + " | ".join(e.index) + " | модель: 10% / медіана / 90% |"
    print(head)
    print("|---|" + "---|" * (len(e) + 1))
    for c in COLS:
        f = FMT[c].format
        cells = []
        for v in e[c]:
            pct = float((ref[c] <= v).mean()) if not pd.isna(v) else np.nan
            mark = " ▲" if pct >= 0.9 else " ▼" if pct <= 0.1 else ""
            cells.append("—" if pd.isna(v) else f"{f(v)}{mark}")
        print(f"| {c} | " + " | ".join(cells) + f" | {f(q.at[0.1, c])} / {f(q.at[0.5, c])} / {f(q.at[0.9, c])} |")
    print("\n▲ — вище, ніж у 90% входів моделі; ▼ — нижче, ніж у 90%.")


if __name__ == "__main__":
    main()
