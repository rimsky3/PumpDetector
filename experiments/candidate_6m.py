"""
experiments/candidate_6m.py — КАНДИДАТ ПРОТИ ПОТОЧНОЇ МОДЕЛІ на 6 місяцях архіву (2026-10-12; власник дорозмітив 5 тікерів і
перенавчив: на 30 днях кандидат слабший, але місяць — малий зріз, тож перевірка на великій вибірці перед «Прийняти»).

Беруться ГОТОВІ моделі з папок (нічого не навчається): 15m-основна, 15m-«старт» і годинна кожного набору, їхні пороги й ознаки
(bundle["features"]). Монети — чесний список 6 міс. (data/long_bin, лише Binance) без тікерів навчання БУДЬ-ЯКОГО з наборів,
тож список однаковий для всіх. Режими: market (вхід на Alert) і split (вхід — Alert «старту», утримання — будь-який Alert).
Вихід — як у роботі (½ на +3 ATR → беззбиток, 4 бари без сигналу, стоп k·ATR); k однаковий для всіх наборів (--k, за
замовчуванням 2.13 — робочий), щоб порівнювати моделі, а не стопи.

Правило проєкту: модель не замінюється, доки нова не обіграє поточну. КРИТЕРІЙ (до запуску), основний режим split, монети
≥ 20 млн: PF кандидата ≥ PF поточної, краще в ≥ 4 з 6 місяців і обидві половини монет не гірші. Режим market — для довідки.
Застереження: обидва набори навчені на розмітці серпня–жовтня, тобто час навчання перекривається з тестом (монети — ні);
це однаково для обох, тож порівняння чесне, а абсолютні PF завищені.
→ reports/candidate_6m.md

    python experiments/candidate_6m.py                                   # models/ проти models/candidate/
    python experiments/candidate_6m.py --sets current=models first=models/candidate_first candidate=models/candidate
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import auto_train as AT
import early_target as ET
import exit_partial as P
import idio_features as I
import train
from fetch import excluded_tickers, read_tickers_file
from tiers import split_scores, tier_frame

BIG = "≥ 20 млн"
OUT = os.path.join("reports", "candidate_6m.md")


def load_set(path: str) -> dict:
    return {"m15": joblib.load(os.path.join(path, train.M15_NAME + ".joblib")),
            "start": joblib.load(os.path.join(path, train.START_NAME + ".joblib")),
            "m1": joblib.load(os.path.join(path, "lightgbm.joblib"))}


def proba(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    return df[["base_asset", "ts"]].assign(probability=bundle["model"].predict_proba(df[bundle["features"]])[:, 1], threshold=bundle["threshold"])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Готові набори моделей на 6 міс. архіву: поточна проти кандидата (market і split)")
    ap.add_argument("--sets", nargs="+", default=["current=models", "candidate=models/candidate"],
                    help="набори назва=папка; перший — поточна модель, останній — кандидат (критерій рахується між ними)")
    ap.add_argument("--k", type=float, default=2.13, help="стоп у ATR(15m), однаковий для всіх наборів (за замовчуванням 2.13)")
    args = ap.parse_args()
    sets = {s.split("=", 1)[0]: load_set(s.split("=", 1)[1]) for s in args.sets}
    uni = json.load(open(os.path.join(I.DIR, "_universe.json"), encoding="utf-8"))
    idx = I.market_index()
    trained = {t.upper() for t in read_tickers_file("tickers.txt")} | set(pd.read_csv("pumps.csv")["base_asset"].str.upper()) | excluded_tickers(".")
    for s in sets.values():
        trained |= set(s["m15"].get("tickers", [])) | set(s["start"].get("tickers", []))
    coins = sorted(c for c in uni if c not in trained and os.path.exists(os.path.join(I.DIR, f"{c}.pkl")))
    start, end = pd.Timestamp(I.START, tz="UTC"), pd.Timestamp(I.END, tz="UTC") + pd.Timedelta(days=1)
    print(f"наборів {len(sets)}: {list(sets)}; монет {len(coins)}; k = {args.k}", flush=True)
    rows = []
    for n, c in enumerate(coins, 1):
        x = pickle.load(open(os.path.join(I.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(I.new_features(h, idx).reindex(f15["ts"]).set_axis(f15.index))
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        for name, s in sets.items():
            share, exit_bars = s["m15"].get("confirm", {}).get("share", 0.75), s["m15"].get("exit_bars", 4)
            s1 = proba(s["m1"], h1)
            win = lambda t: t.loc[(t.index >= start) & (t.index < end), "alert_score"]      # noqa: E731
            main_s = win(tier_frame(proba(s["m15"], f15), s1, share).set_index("ts").sort_index())
            start_s = win(tier_frame(proba(s["start"], f15), s1, share).set_index("ts").sort_index())
            entry, hold = split_scores(main_s, start_s)
            for mode, trades in (("market", P.simulate(hh, main_s, args.k, exit_bars, 3.0, 0.5, True)),
                                 ("split", ET.simulate_split(hh, entry, hold, args.k, exit_bars, 3.0, 0.5, True))):
                for r in trades:
                    mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                    if mth in uni[c]:
                        rows.append({"set": name, "mode": mode, "ticker": c, "half": AT.half(c), "month": mth, "band": uni[c][mth],
                                     "ret": r["ret"], "reason": r["reason"]})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    d = pd.DataFrame(rows)
    pf = I.pf
    names = list(sets)
    L = ["# Кандидат проти поточної моделі на 6 місяцях (12.10)", "",
         f"Готові моделі з папок; {len(coins)} монет чесного списку поза навчанням усіх наборів; стоп k = {args.k} для всіх. "
         "Час навчання перекривається з тестом однаково для всіх наборів — абсолютні PF завищені, порівняння чесне.", ""]
    for name, s in sets.items():
        L.append(f"- **{name}**: 15m навчено {str(s['m15'].get('trained_at'))[:16]}, тікерів {len(s['m15'].get('tickers', []))}, поріг "
                 f"{s['m15']['threshold']:.3f}; «старт» поріг {s['start']['threshold']:.3f}; годинна поріг {s['m1']['threshold']:.3f}")
    res = {}
    for mode in ("split", "market"):
        L += ["", f"## Режим {mode}", "", "| Набір | Угод | PF усі | Угод ≥ 20 млн | PF ≥ 20 млн | без топ-3 | У плюсі | По стопу | A | B | "
              + " | ".join(m[5:] for m in sorted(d["month"].unique())) + " |", "|---|" + "---|" * (9 + d["month"].nunique())]
        for name in names:
            g = d[(d["set"] == name) & (d["mode"] == mode)]
            b = g[g["band"] == BIG]
            months = {m: pf(x["ret"]) for m, x in b.groupby("month")}
            res[(mode, name)] = {"pf20": pf(b["ret"]), "A": pf(b[b["half"] == "A"]["ret"]), "B": pf(b[b["half"] == "B"]["ret"]), "months": months}
            L.append(f"| {name} | {len(g)} | {pf(g['ret']):.3f} | {len(b)} | **{pf(b['ret']):.3f}** | {pf(b['ret'].sort_values().iloc[:-3]):.3f} | "
                     f"{(b['ret'] > 0).mean():.0%} | {(b['reason'] == 'stop').mean():.0%} | {res[(mode, name)]['A']:.3f} | {res[(mode, name)]['B']:.3f} | "
                     + " | ".join(f"{months.get(m, float('nan')):.2f}" for m in sorted(d["month"].unique())) + " |")
    cur, cand = res[("split", names[0])], res[("split", names[-1])]
    better = sum(cand["months"].get(m, np.nan) > cur["months"].get(m, np.nan) for m in cur["months"])
    ok = cand["pf20"] >= cur["pf20"] and better >= 4 and cand["A"] >= cur["A"] and cand["B"] >= cur["B"]
    mc, mk = res[("market", names[0])], res[("market", names[-1])]
    L += ["", f"**Критерій (split, ≥ 20 млн: {names[-1]} ≥ {names[0]}, краще в ≥ 4 з 6 місяців, обидві половини не гірші): "
          f"{'ВИКОНАНО' if ok else 'НЕ ВИКОНАНО'}** — PF {cand['pf20']:.3f} проти {cur['pf20']:.3f}; краще в {better} з {len(cur['months'])} місяців; "
          f"A {cand['A']:.3f} / {cur['A']:.3f}, B {cand['B']:.3f} / {cur['B']:.3f}.",
          f"Довідково, market: {mk['pf20']:.3f} проти {mc['pf20']:.3f}."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
