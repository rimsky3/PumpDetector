"""
experiments/stop_size.py — менший стоп, ніж у моделі: 6 місяців, чесний список монет (2026-10-08).

Питання власника: «я торгую Alert руками і ставлю стопи менші, ніж модель (буває стоп 10%), а якщо в стакані видно, що
імпульсу не буде, — виходжу раніше. Що буде з бектестом, якщо стоп, наприклад, на 50% менший?»

Входи — Alert робочої моделі (як exit_5m.py: 15m з власним рухом + годинна, монети чесного списку станом на кожен місяць,
квітень–вересень, лише Binance). Позиція як у дашборді, перевірки на 5m-барах (experiments/exit_5m.sim5): ½ на +3 ATR і
стоп у беззбиток, решта — 4 бари без Alert або стоп. Змінюється лише стоп: k × {100%, 75%, 50%, 35%}. Після стопу —
або новий вхід на наступному барі з Alert (як зараз), або «чекати, доки Alert згасне» (ближче до ручної торгівлі: вибило —
у той самий рух не лізеш). Ціль +3 ATR — та сама (не прив'язана до стопу).

Чого тут немає: виходу «за стаканом» — його з наших даних не змоделювати; тест показує лише, що дає сам менший стоп.

    python experiments/stop_size.py      # → reports/stop_size.md (5m-кеш з exit_5m.py: data/exit5m/)
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import exit_5m as X5
import idio_features as E
import train
from features import move_features
from tiers import tier_frame

OUT = os.path.join("reports", "stop_size.md")
MULTS = (1.0, 0.75, 0.5, 0.35)


def pf(v) -> float:
    v = pd.Series(v, dtype=float)
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    argparse.ArgumentParser(description="Менший стоп, ніж у моделі (k × 100/75/50/35%): 6 міс., чесний список, 5m-бари").parse_args()
    coins, uni = X5.coins_list()
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path(m15.get("confirm", {}).get("model", "lightgbm")))
    k, exit_bars, share = m15["risk"]["k"], m15.get("exit_bars", 4), m15.get("confirm", {}).get("share", 0.75)
    variants = {}
    for m in MULTS:
        variants[f"{int(m * 100)}%"] = dict(k=k * m)
        variants[f"{int(m * 100)}% · чекати"] = dict(k=k * m, wait=True, wait_on=("stop",))
    print(f"модель {m15['trained_at']}, k {k:.2f}; {len(coins)} монет; індекс ALT-EW з кешу…", flush=True)
    idx = E.market_index()
    start, end = pd.Timestamp(X5.START, tz="UTC"), pd.Timestamp(X5.END, tz="UTC") + pd.Timedelta(days=1)
    rows = []
    for n, c in enumerate(coins, 1):
        h5 = X5.load_5m(c)
        if h5 is None or h5.empty:
            continue
        x = pickle.load(open(os.path.join(X5.DIR, f"{c}.pkl"), "rb"))
        f15, h1, h = x["f15"], x["h1"], x["h"]
        f15 = f15.join(move_features(h, idx.reindex(h.index), 4).reindex(f15["ts"]).set_axis(f15.index))
        s15 = f15[["base_asset", "ts"]].assign(probability=m15["model"].predict_proba(f15[m15["features"]])[:, 1], threshold=m15["threshold"])
        s1 = h1[["base_asset", "ts"]].assign(probability=m1["model"].predict_proba(h1[m1["features"]])[:, 1], threshold=m1["threshold"])
        t = tier_frame(s15, s1, share).set_index("ts").sort_index()
        sc = t.loc[(t.index >= start) & (t.index < end), "alert_score"]
        hh = h[(h.index >= start - pd.Timedelta(days=1)) & (h.index < end)]
        months = uni[c]
        for name, kw in variants.items():
            kw = dict(kw)
            kk = kw.pop("k")
            for r in X5.sim5(hh, sc, h5, kk, exit_bars, **kw):
                mth = pd.Timestamp(r["ts"]).strftime("%Y-%m")
                if mth in months:
                    rows.append({"variant": name, "ticker": c, "month": mth, "band": months[mth], **r})
        if n % 50 == 0:
            print(f"  {n}/{len(coins)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join("reports", "stop_size_trades.csv"), index=False)
    names = list(variants)
    lines = ["# Менший стоп, ніж у моделі: 6 місяців, чесний список монет", "",
             f"{tr['ticker'].nunique()} монет поза навчанням (список станом на кожен місяць), {X5.START} – {X5.END}, лише Binance. Входи — "
             f"Alert робочої моделі {m15['trained_at']}; стоп = k × множник (k = {k:.2f} ATR15), ½ на +3 ATR і б/з, решта — {exit_bars} бари "
             "без Alert; перевірки на 5m-барах; комісія 0.1%. «чекати» — після стопу не заходити знову, доки Alert не згасне.", "",
             "| стоп | угод | стоп, % (медіана) | у плюсі | медіана | PF | PF без топ-3 | ≥ 20 млн | 5–20 млн | по стопу | ½ взято | сер. програш | сер. виграш |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in names:
        g = tr[tr["variant"] == v]
        r = g["ret"].sort_values()
        lines.append(f"| {v} | {len(r)} | {g['stop_pct'].median():.1%} | {(r > 0).mean():.0%} | {r.median():+.2%} | {pf(r):.2f} | "
                     f"{pf(r.iloc[:-3]):.2f} | {pf(g[g['band'] == '≥ 20 млн']['ret']):.2f} | {pf(g[g['band'] == '5–20 млн']['ret']):.2f} | "
                     f"{(g['reason'] == 'stop').mean():.0%} | {g['took'].mean():.0%} | {r[r <= 0].mean():+.2%} | {r[r > 0].mean():+.2%} |")
    piv = tr.groupby(["month", "variant"])["ret"].apply(pf).unstack()[names]
    lines += ["", "## По місяцях (PF)", "", "| місяць | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m, r in piv.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{r[v]:.2f}" for v in names) + " |")
    lines += ["", "Місяців, де варіант кращий за 100%: " + ", ".join(f"{v} {int((piv[v] > piv['100%'] + 1e-9).sum())} з {len(piv)}"
                                                         for v in names if v != "100%"), ""]
    # розмір стопу в % — як часто стоп «великий» (10%+)
    g = tr[tr["variant"] == "100%"]
    lines += ["## Розмір стопу моделі зараз (100%)", "",
              f"Медіана {g['stop_pct'].median():.1%}; ≥ 5% — {(g['stop_pct'] <= -0.05).mean():.0%} угод, ≥ 10% — {(g['stop_pct'] <= -0.10).mean():.0%}.", ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Звіт: {OUT}")


if __name__ == "__main__":
    main()
