"""
experiments/p_exit.py — мітка p («залочка» на хаях) як ВИХІД з лонгу і як фільтр входу (2026-10-07).

Власник (PONS 06.10 18:15 / 07.10 16:00): після p ціна йшла вниз; модель увійшла о 18:00, p на наступному барі, а вихід
моделі — через 4 бари з −2.5%. Як самостійний шорт p з архіву поки ≈ 0R (25 подій, +0.02R), тож перевіряємо інше
використання — правило в позиції моделі:
    base        як зараз: ½ на +3 ATR (далі б/з), решта — 4 бари без Alert або стоп k·ATR
    exit_p      + вихід усієї (залишку) позиції на закритті бару, де з'явилась p
    be_p        + на p стоп переноситься в беззбиток (без виходу)
    filter      не входити, якщо p була на барі сигналу або барі перед ним
    filter+exit фільтр і вихід на p разом
    exit_p_wait вихід на p і БЕЗ повторного входу, доки Alert не згасне (новий вхід — лише в новому епізоді): у exit_p після
                виходу модель одразу купувала ту саму вершину знову (+300 угод)
    be_p_wait   стоп у б/з на p; після виходу з такої угоди — так само без повторного входу в тому ж епізоді
Мітки p — footprint.patterns (те саме правило, що наживо), лише в контексті (Кандидат / Alert за 6 год), на футпрінті з
архіву угод Binance (кеш data/flow, з 28.08). Сигнали й рівні — backtest.tiers_and_bars (робоча модель). Монети —
бектесту, поза навчанням. Крім загальних метрик — ПОПАРНО на угодах, де p з'являлась (різниця для тих самих угод) і
по половинах періоду.

    python experiments/p_exit.py [--days 40]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import footprint as FP
import train
from backtest import tiers_and_bars
from exit_partial import FEE
from features import load_pumps
from fetch import excluded_tickers, read_tickers_file
from spring_pattern import CACHE, load_coin

OUT = os.path.join("reports", "p_exit.md")
VARIANTS = ["base", "exit_p", "be_p", "filter", "filter+exit", "exit_p_wait", "be_p_wait"]


def simulate(b: pd.DataFrame, score: np.ndarray, p: np.ndarray, sl: float, exit_bars: int, variant: str,
             tp: float = 3.0, frac: float = 0.5) -> list[dict]:
    """Як exit_partial.simulate (½ на +tp ATR, далі б/з; решта — exit_bars без сигналу або стоп), плюс правила для p."""
    a, hi, lo, c = b["atr"].to_numpy(), b["high"].to_numpy(), b["low"].to_numpy(), b["price"].to_numpy()
    out, tr, below, wait = [], None, 0, False
    use_exit = variant in ("exit_p", "filter+exit", "exit_p_wait")
    use_be, use_filter = variant in ("be_p", "be_p_wait"), variant in ("filter", "filter+exit")
    use_wait = variant.endswith("_wait")
    for i in range(len(b)):
        s = score[i]
        if tr is None:
            if wait and (np.isnan(s) or s < 1):
                wait = False                 # епізод Alert скінчився — наступний сигнал уже новий
            if wait or np.isnan(s) or s < 1 or np.isnan(c[i]) or np.isnan(a[i]):
                continue
            if use_filter and (p[i] or (i > 0 and p[i - 1])):
                continue
            tr = {"e": c[i], "stop": c[i] - sl * a[i], "tp": c[i] + tp * a[i], "part": 0.0, "i0": i, "saw_p": False}
            below = 0
            continue
        e = tr["e"]
        if lo[i] <= tr["stop"]:
            rest = tr["stop"] / e - 1
            out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - FEE, "reason": "stop",
                        "ts": b.index[tr["i0"]], "saw_p": tr["saw_p"]})
            wait = use_wait and tr["saw_p"]
            tr = None
            continue
        if not tr["part"] and hi[i] >= tr["tp"]:
            tr["part"] = frac * (tr["tp"] / e - 1)
            tr["stop"] = max(tr["stop"], e)
        if p[i]:
            tr["saw_p"] = True
            if use_exit:
                rest = c[i] / e - 1
                out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - FEE, "reason": "p",
                            "ts": b.index[tr["i0"]], "saw_p": True})
                wait = use_wait
                tr = None
                continue
            if use_be:
                tr["stop"] = max(tr["stop"], e)
        below = below + 1 if (np.isnan(s) or s < 1) else 0
        if below >= exit_bars:
            rest = c[i] / e - 1
            out.append({"ret": tr["part"] + (1 - (frac if tr["part"] else 0)) * rest - FEE, "reason": "model",
                        "ts": b.index[tr["i0"]], "saw_p": tr["saw_p"]})
            tr = None
    return out


def pf(v) -> float:
    v = pd.Series(v)
    gl = -v[v <= 0].sum()
    return v[v > 0].sum() / gl if gl else np.nan


def stats(r: pd.Series) -> dict:
    s = r.sort_values()
    return {"угод": len(s), "у плюсі": f"{(s > 0).mean():.0%}", "медіана": f"{s.median():+.2%}", "сума": f"{s.sum():+.0%}",
            "PF": round(pf(s), 2), "PF без топ-3": round(pf(s.iloc[:-3]), 2), "≥ +10%": int((s >= 0.10).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Мітка p як вихід з лонгу моделі і як фільтр входу (монети бектесту, архів угод)")
    ap.add_argument("--days", type=int, default=40, help="скільки днів (кеш архіву угод — з 28.08)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    m15 = joblib.load(train.model_path(train.M15_NAME))
    m1 = joblib.load(train.model_path("lightgbm"))
    k, exit_bars = m15.get("risk", {}).get("k", 2.48), m15.get("exit_bars", 4)
    coins = set()
    for path in glob.glob(os.path.join("reports", "backtests", "2026*_alert.json")):
        coins |= {u["base"] for u in json.load(open(path, encoding="utf-8"))["meta"]["universe"]}
    rows, n_p = [], 0
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        trained = {t.upper() for t in read_tickers_file("tickers.txt")} | set(load_pumps(conn)["base_asset"]) | excluded_tickers(".")
        coins = sorted((coins - trained) & set(os.listdir(CACHE)))
        print(f"{len(coins)} монет бектесту з архівом угод; k = {k:.2f}", flush=True)
        for n, base in enumerate(coins, 1):
            x = load_coin(conn, base)
            th = tiers_and_bars(conn, base, m15, m1, args.days) if x is not None else None
            if th is None:
                continue
            t = th[0]
            b = FP.bar_metrics(*x)
            b = b[b.index >= t.index.min()]
            tier = t["tier"].reindex(b.index)
            ctx = tier.isin(["candidate", "alert"]).astype(int).rolling(FP.CTX_BARS, min_periods=1).max().astype(bool)
            p = (FP.patterns(b)["p"] & ctx).to_numpy()
            n_p += int(p.sum())
            score = t["alert_score"].reindex(b.index).to_numpy()
            for v in VARIANTS:
                rows += [{**r, "variant": v, "ticker": base} for r in simulate(b, score, p, k, exit_bars, v)]
            if n % 40 == 0:
                print(f"  {n}/{len(coins)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join("reports", "p_exit_trades.csv"), index=False)
    tot = pd.DataFrame({v: stats(df[df["variant"] == v]["ret"]) for v in VARIANTS}).T
    # попарно: угоди base, у яких була p, проти тих самих угод (той самий вхід) в exit_p / be_p
    key = ["ticker", "ts"]
    base = df[df["variant"] == "base"].set_index(key)
    pair_rows = {}
    for v in ("exit_p", "be_p", "exit_p_wait", "be_p_wait"):
        other = df[df["variant"] == v].set_index(key)
        j = base[base["saw_p"]][["ret"]].join(other[["ret"]], rsuffix="_v", how="inner")
        d = j["ret_v"] - j["ret"]
        pair_rows[v] = {"угод з p": len(j), "base сума": f"{j['ret'].sum():+.0%}", f"{v} сума": f"{j['ret_v'].sum():+.0%}",
                        "різниця медіана": f"{d.median():+.2%}", "краще / гірше / так само": f"{(d > 1e-9).sum()} / {(d < -1e-9).sum()} / {(d.abs() <= 1e-9).sum()}",
                        "base ≥ +10% обрізано": int(((j["ret"] >= 0.10) & (j["ret_v"] < 0.10)).sum())}
    pairs = pd.DataFrame(pair_rows).T
    mid = df["ts"].min() + (df["ts"].max() - df["ts"].min()) / 2
    halves = pd.DataFrame({f"{v} · {'1-ша' if h == 0 else '2-га'} пол.": stats(df[(df["variant"] == v) & ((df["ts"] >= mid) == bool(h))]["ret"])
                           for v in VARIANTS for h in (0, 1)}).T[["угод", "PF", "PF без топ-3", "медіана"]]
    md = lambda x: "```\n" + x.to_string() + "\n```"   # noqa: E731
    lines = ["# Мітка p як вихід з лонгу / фільтр входу", "",
             f"{df['ticker'].nunique()} монет бектесту поза навчанням, {args.days} днів (архів угод Binance), міток p у контексті: {n_p}; "
             f"частковий вихід ½ на +3 ATR, стоп {k:.2f} ATR, комісія 0.1%.", "",
             "## Загалом", "", md(tot), "", "## Попарно — лише угоди, де з'являлась p", "", md(pairs), "",
             "## По половинах періоду", "", md(halves), ""]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print(f"Звіт: {OUT}")


if __name__ == "__main__":
    main()
