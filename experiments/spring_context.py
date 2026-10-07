"""
experiments/spring_context.py — «прокол лоу з викупом» лише в контексті моделі (2026-10-07).

experiments/spring_pattern.py: механічний прокол з викупом на всіх барах переваги не має (≈ випадковий бар). Власник
торгує його не будь-де, а в монеті «в грі» — на відкаті всередині імпульсу (ORCA 06.10 20:00) або перед стартом, і
бачить підтвердження наступними барами. Тут — саме ця умова, взята з того, як він торгує (не підбір порогів):
  контекст  — за останні 6 год (24 бари, включно з поточним) був Кандидат або Alert робочої моделі;
  A         — прокол лоу 2 год з викупом (правило spring_pattern: ≥ 30% обсягу під лоу, закриття > лоу, верхня третина
              ≤ 20%, обсяг ≥ 2×), вхід на закритті бару, стоп під лоу бару;
  A + підтв. — наступний бар: дельта > 0 і закриття вище хаю бару проколу; вхід на закритті наступного бару, стоп той самий;
  контроль  — тінь під лоу в контексті (прокол із закриттям > лоу, без умов на обсяг і футпрінт), те саме з підтвердженням,
              випадкові бари контексту, і ВХІД МОДЕЛІ (перший бар Alert епізоду; стоп k·ATR з бандла і, для порівняння R,
              під лоу бару входу).
Монети — лише бектесту (модель їх не бачила); оцінки — backtest.tiers_and_bars (робоча 15m + годинна з власних 15m).
Метрики — як spring_pattern: ціна через 1 / 4 / 12 год, +1R / +2R раніше за стоп (12 год), середній R (вихід +2R / стоп /
12 год, комісія 0.1%).

    python experiments/spring_context.py [--days 40]
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import train
from backtest import tiers_and_bars
from features import load_pumps
from fetch import read_tickers_file
from spring_pattern import CACHE, FEE, HORIZON, bar_metrics, load_coin, summary

pd.DataFrame.to_markdown = lambda self, *a, **k: "```\n" + self.to_string() + "\n```"
OUT = os.path.join("reports", "spring_context.md")
CTX_BARS = 24


def trade(b: pd.DataFrame, i: int, stop: float, base: str, kind: str) -> dict | None:
    """Вхід на закритті бару i, стоп stop; ціна через 1/4/12 год і R (вихід +2R / стоп / 12 год)."""
    c, hi, lo = b["price"].to_numpy(), b["high"].to_numpy(), b["low"].to_numpy()
    if i + HORIZON >= len(b) or not (stop < c[i]):
        return None
    R = c[i] - stop
    r = {f"ret_{h}h": c[i + 4 * h] / c[i] - 1 for h in (1, 4, 12)}
    r["ret_4h_atr"] = (c[i + 16] - c[i]) / b["atr"].iloc[i]
    res, hit1, hit2 = None, False, False
    for j in range(i + 1, i + HORIZON + 1):
        if lo[j] <= stop:
            res = -1.0
            break
        hit1 |= hi[j] - c[i] >= R
        if hi[j] - c[i] >= 2 * R:
            hit2, res = True, 2.0
            break
    if res is None:
        res = (c[i + HORIZON] - c[i]) / R
    r.update({"hit1R": hit1 or hit2, "hit2R": hit2, "R": res - FEE * c[i] / R, "R_pct": R / c[i], "base": base,
              "ts": b.index[i], "kind": kind})
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description="Прокол лоу з викупом лише в контексті Кандидата / Alert моделі (монети бектесту)")
    ap.add_argument("--days", type=int, default=40, help="скільки днів оцінювати (кеш архіву — з 28.08)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    pd.set_option("display.width", 250)
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    k = m15.get("risk", {}).get("k", 2.48)
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    coins = set()
    for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
        coins |= {u["base"] for u in json.load(open(path, encoding="utf-8"))["meta"]["universe"]}
    rows, n_ctx, n_bars = [], 0, 0
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        trained = reviewed | set(load_pumps(conn)["base_asset"])
        coins = sorted((coins - trained) & set(os.listdir(CACHE)))
        print(f"{len(coins)} монет бектесту з архівом; k = {k:.2f}", flush=True)
        for n, base in enumerate(coins, 1):
            x = load_coin(conn, base)
            th = tiers_and_bars(conn, base, m15, m1, args.days) if x is not None else None
            if th is None:
                continue
            b = bar_metrics(*x)
            t = th[0]
            b = b.join(t[["tier"]], how="left")
            b = b[b.index >= t.index.min()]
            sig = b["tier"].isin(["candidate", "alert"]).astype(int)
            alert = (b["tier"] == "alert").astype(int)
            ctx = sig.rolling(CTX_BARS, min_periods=1).max().astype(bool)
            ok = b["L"].notna() & b["vol_ratio"].notna() & b["atr"].notna()
            wick = (b["low"] < b["L"]) & (b["price"] > b["L"]) & ok
            A = wick & (b["below_share"] >= 0.3) & (b["top_share"] <= 0.2) & (b["vol_ratio"] >= 2)
            nxt_ok = (b["delta"].shift(-1) > 0) & (b["price"].shift(-1) > b["high"])
            first_alert = (alert == 1) & (alert.shift(1, fill_value=0).rolling(4, min_periods=1).max() == 0)
            n_ctx += int(ctx.sum()); n_bars += len(b)
            idx = {name: np.flatnonzero(m.to_numpy()) for name, m in {
                "A в контексті": A & ctx, "A в контексті + підтв.": A & ctx & nxt_ok,
                "тінь під лоу в контексті": wick & ctx, "тінь в контексті + підтв.": wick & ctx & nxt_ok,
                "A поза контекстом": A & ~ctx,
                "випадкові бари контексту": ctx & ok & (np.arange(len(b)) % 8 == 0),
                "вхід моделі (Alert), стоп k·ATR": first_alert & ok, "вхід моделі (Alert), стоп під лоу бару": first_alert & ok}.items()}
            lows = b["low"].to_numpy()
            for name, ii in idx.items():
                for i in ii:
                    if "підтв." in name:
                        r = trade(b, i + 1, lows[i] * 0.999, base, name)          # вхід на наступному барі, стоп під лоу проколу
                    elif name.endswith("k·ATR"):
                        r = trade(b, i, b["price"].iloc[i] - k * b["atr"].iloc[i], base, name)
                    else:
                        r = trade(b, i, lows[i] * 0.999, base, name)
                    if r:
                        rows.append(r)
            if n % 40 == 0:
                print(f"  {n}/{len(coins)}", flush=True)
    df = pd.DataFrame(rows)
    order = ["вхід моделі (Alert), стоп k·ATR", "вхід моделі (Alert), стоп під лоу бару", "A в контексті", "A в контексті + підтв.",
             "тінь під лоу в контексті", "тінь в контексті + підтв.", "випадкові бари контексту", "A поза контекстом"]
    s = pd.DataFrame({kname: summary(df[df["kind"] == kname]) for kname in order}).T
    print(f"\nКонтекст (Кандидат / Alert за 6 год): {n_ctx / max(n_bars, 1):.0%} барів\n" + s.to_string(), flush=True)
    # стійкість: перша і друга половина періоду
    mid = df["ts"].min() + (df["ts"].max() - df["ts"].min()) / 2
    halves = pd.DataFrame({f"{kname} · {'1-ша' if h == 0 else '2-га'} пол.": summary(df[(df["kind"] == kname) & ((df["ts"] >= mid) == bool(h))])
                           for kname in order[:6] for h in (0, 1)}).T[["подій", "4 год", "12 год", "+2R до стопу", "R сер."]]
    print("\nПо половинах періоду:\n" + halves.to_string(), flush=True)
    lines = [f"# Прокол лоу з викупом у контексті моделі ({datetime.now(timezone.utc):%Y-%m-%d})", "",
             f"{len(coins)} монет бектесту (поза навчанням), {args.days} днів; контекст — {n_ctx / max(n_bars, 1):.0%} барів.", "",
             s.to_markdown(), "", "## По половинах періоду", "", halves.to_markdown(), ""]
    os.makedirs("reports", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    df.to_csv(os.path.join("reports", "spring_context_events.csv"), index=False)
    print(f"\nзвіт: {OUT}")


if __name__ == "__main__":
    main()
