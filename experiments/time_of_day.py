"""
experiments/time_of_day.py — у які години доби й дні тижня модель працює найкраще, коли більше збиткових Alert і стопів (2026-10-03).

Угоди — поточне правило (вхід по ринку на Alert, стоп k·ATR, ½ на +3 ATR, далі б/з; решта — 4 бари без Alert):
    6 міс. — кеш pullback_grid (data/long_bt, Bybit 15m, 205 монет поза навчанням, оцінки моделі 30.09) — велика вибірка;
    вер — бектест 03.10 новою моделлю (k 2.48), монети ≥ 20 млн $ (reports/backtests/20261003_1216_…) — перевірка.
Час — вхід (закриття сигнального бару), UTC і Київ (UTC+3 у вересні–жовтні). «Збиткова» — результат ≤ 0 з комісією.

    python experiments/time_of_day.py     # → reports/time_of_day.md
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import pullback_grid as PG

K = 2.480727396398431
RUN = "20261003_1216_candidate_market_k2.48"
OUT = os.path.join("reports", "time_of_day.md")
KYIV = 3
BLOCKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]


def pf(v: pd.Series) -> float:
    gl = -v[v <= 0].sum()
    return round(v[v > 0].sum() / gl, 2) if gl else np.nan


def stats(g: pd.DataFrame, days: float) -> dict:
    v = g["ret"]
    return {"угод": len(g), "на день": round(len(g) / days, 1), "у плюсі": f"{(v > 0).mean():.0%}", "PF": pf(v),
            "медіана": f"{v.median():+.2%}", "збиткових": f"{(v <= 0).mean():.0%}", "по стопу": f"{(g['why'] == 'стоп').mean():.0%}",
            "≥ +10%": int((v >= 0.10).sum())}


def table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def report(df: pd.DataFrame, title: str) -> list[str]:
    days = (df["dt"].max() - df["dt"].min()).total_seconds() / 86400 or 1
    md = [f"## {title}", "", f"{len(df)} угод, {df['base'].nunique()} монет, {days:.0f} днів. Загалом: PF {pf(df['ret'])}, у плюсі "
          f"{(df['ret'] > 0).mean():.0%}, по стопу {(df['why'] == 'стоп').mean():.0%}.", ""]
    rows = []
    for a, b in BLOCKS:
        g = df[(df["hour"] >= a) & (df["hour"] < b)]
        rows.append({"години UTC": f"{a:02d}–{b:02d}", "Київ": f"{(a + KYIV) % 24:02d}–{(b + KYIV) % 24:02d}", **stats(g, days)})
    md += ["### За 4-годинними блоками", "", table(rows), ""]
    rows = [{"година UTC": f"{h:02d}", "Київ": f"{(h + KYIV) % 24:02d}", **stats(df[df["hour"] == h], days)} for h in range(24)]
    md += ["### За годинами (шумніше)", "", table(rows), ""]
    names = ["пн", "вт", "ср", "чт", "пт", "сб", "нд"]
    rows = [{"день": names[d], **stats(df[df["dow"] == d], days / 7)} for d in range(7)]
    md += ["### За днями тижня (UTC; «на день» — на один такий день)", "", table(rows), ""]
    return md


def main() -> None:
    argparse.ArgumentParser(description="Результат угод моделі за годинами доби й днями тижня (6 міс. + вересень новою моделлю).").parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    data = pickle.load(open(PG.CACHE, "rb"))
    rows = []
    for b, A in data["6міс"].items():
        for i, ts, res, why, wait in PG.simulate(A, "market", 0.0, 0, "atr", K):
            if res is not None:
                rows.append({"base": b, "ts": ts, "ret": res, "why": why})
    six = pd.DataFrame(rows)
    six["dt"] = pd.to_datetime(six["ts"], utc=True) + pd.Timedelta(minutes=15)       # вхід — закриття сигнального бару
    d = json.load(open(os.path.join("reports", "backtests", RUN + ".json"), encoding="utf-8"))
    sep = pd.DataFrame([{"base": t["ticker"], "dt": pd.Timestamp(t["entry_ts"], unit="s", tz="UTC") + pd.Timedelta(minutes=15),
                         "ret": t["ret"], "why": "стоп" if t["reason"] == "stop" else t["reason"]}
                        for t in d["trades_list"] if t["reason"] != "open" and t["ret"] == t["ret"]])
    md = ["# Коли модель працює найкраще — години доби й дні тижня", "",
          "Вхід по ринку на Alert, стоп 2.48 ATR, частковий вихід (½ на +3 ATR, б/з), комісія 0.1%. Час — вхід (закриття сигнального "
          "бару); Київ = UTC+3. «Збиткових» — результат ≤ 0 з комісією; «по стопу» — закрито стопом до цілі ½.", ""]
    for df, title in ((six, "6 місяців (квіт–вер), Bybit, оцінки моделі 30.09, 205 монет поза навчанням"),
                      (sep, "Вересень, нова модель 03.10, монети ≥ 20 млн $ поза навчанням")):
        df["hour"], df["dow"] = df["dt"].dt.hour, df["dt"].dt.dayofweek
        md += report(df, title)
    open(OUT, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
