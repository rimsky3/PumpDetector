"""
experiments/spring_pattern.py — «прокол лоу з викупом» (форма b) і «кульмінація + розворот дельти» з футпрінту (2026-10-07).

Звідки: власник торгує патерн з atas.net («хибний пробій зі стопами»): ціна проколює попередній локальний лоу, там
великий обсяг (стопи, ліквідації, шорти на пробій), але ціна не падає далі — великий покупець стоїть лімітками; бар
закривається вгорі, обсяг унизу бару (форма b). Приклади власника (угоди Binance, 15m): ORCA 06.10 20:00, API3 05.10 14:30,
CAP 06.10 03:15 — прокол лоу 2 год на 0.9–2.9%, під лоу 34–86% обсягу бару, верхня третина 14–17%; CAP 06.10 14:15 —
інший патерн: після зливу на величезному обсязі дельта стала додатною, а нового низу немає.

Правила (зафіксовані ДО прогону, 15m-бари Binance; лоу/хай — попередні 2 год = 8 барів):
  A  прокол з викупом: лоу бару < L; під L ≥ 30% обсягу бару; закриття > L; верхня третина діапазону ≤ 20% обсягу;
     обсяг бару ≥ 2× медіани за попередню добу. Дельта під L — лише описово.
  B  кульмінація + розворот дельти: два попередні бари разом впали ≥ 2 ATR, обсяг кожного ≥ 3× медіани, дельта < 0;
     бар з дельтою > 3% свого обсягу і лоу не нижче за лоу зливу − 1 ATR.
  U  дзеркальний A на хаях (прокол хаю з викупом продавцем): хай > H; над H ≥ 30% обсягу; закриття < H; нижня третина ≤ 20%;
     обсяг ≥ 2×.
Контроль: усі бари; будь-який прокол лоу; прокол з закриттям вище L (тінь, без умов на обсяг); пробій (закриття нижче L).

Що міряємо від закриття сигнального бару: ціна через 1 / 4 / 12 год (%, і в ATR), частка > 0 за 4 год; угода «вхід на
закритті, стоп під лоу бару (A, B) / над хаєм (U, шорт)»: дійшло до +1R / +2R раніше за стоп (12 год), середній R при
виході «+2R / стоп / 12 год» з комісією 0.1%. Плюс розмітка: як часто A є за ≤ 3 год до старту імпульсу проти звичайного
3-год вікна, і U в першу годину фальстарту проти імпульсу.

Дані: кеш data/flow/ (experiments/flow_archive.py: 15m-зведення і футпрінт угод Binance на сітці 0.1%) + OHLC 15m Binance з бази.

    python experiments/spring_pattern.py
"""

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

import footprint as FP
from features import load_pumps, query_df
from fetch import read_tickers_file
from footprint import FEE, HORIZON, bar_metrics

pd.DataFrame.to_markdown = lambda self, *a, **k: "```\n" + self.to_string() + "\n```"

CACHE = os.path.join("data", "flow")
OUT = os.path.join("reports", "spring_pattern.md")


def load_coin(conn, base: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    days = sorted(glob.glob(os.path.join(CACHE, base, "*.pkl")))
    parts = [x for x in (pd.read_pickle(p) for p in days) if x is not None and len(x["bars"])]
    if not parts:
        return None
    flow = pd.concat([x["bars"] for x in parts]).sort_index()
    fp = pd.concat([x["fp"] for x in parts], ignore_index=True)
    o = query_df(conn, """
        SELECT t.ts, t.o, t.h, t.l, t.c FROM ohlcv t JOIN markets m USING (symbol)
        WHERE t.interval = '15min' AND m.base_asset = %s AND m.exchange_name = 'Binance' AND m.quote_asset = 'USDT'
        ORDER BY t.ts""", (base,))
    if o.empty:
        return None
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    o = o.drop_duplicates("ts").set_index("ts")
    b = o.join(flow[["v_buy", "v_sell"]], how="inner")
    b = b.rename(columns={"c": "price", "h": "high", "l": "low", "o": "open"})
    return b, fp[fp["ts"].isin(b.index)]


def events(b: pd.DataFrame) -> dict[str, pd.Series]:
    """A / U — правило footprint.patterns (b / p); B і контрольні групи — лише для цього експерименту."""
    pat = FP.patterns(b)
    pierce = b["low"] < b["L"]
    fall = (b["price"].shift(1) - b["open"].shift(2)) <= -2 * b["atr"].shift(1)
    climax = fall & (b["vol_ratio"].shift(1) >= 3) & (b["vol_ratio"].shift(2) >= 3) & ((b["delta"].shift(1) + b["delta"].shift(2)) < 0)
    flush_low = pd.concat([b["low"].shift(1), b["low"].shift(2)], axis=1).min(axis=1)
    B = climax & (b["delta"] > 0.03 * b["vol"]) & (b["low"] >= flush_low - b["atr"])
    ok = b["L"].notna() & b["vol_ratio"].notna() & b["atr"].notna()
    return {"A прокол з викупом": pat["b"] & ok, "B кульмінація + дельта": B & ok, "U прокол хаю (шорт)": pat["p"] & ok,
            "к: усі бари": ok, "к: будь-який прокол лоу": pierce & ok,
            "к: тінь під лоу (закриття > L)": pierce & (b["price"] > b["L"]) & ok, "к: пробій (закриття < L)": pierce & (b["price"] <= b["L"]) & ok}


def outcomes(b: pd.DataFrame, mask: pd.Series, short: bool, base: str) -> pd.DataFrame:
    """Угода від закриття кожного бару маски: стоп під лоу (над хаєм для шорту), footprint.outcome; лише повний горизонт."""
    rows = []
    for i in np.flatnonzero(mask.to_numpy()):
        if i + HORIZON >= len(b):
            continue
        r = FP.outcome(b, i, FP.stop_of(b, i, "p" if short else "b"), short=short)
        if r:
            rows.append(r | {"base": base, "ts": b.index[i], "below_delta": b["below_delta"].iloc[i]})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"подій": 0}
    return {"подій": len(df), "монет": df["base"].nunique(), "1 год": f"{df['ret_1h'].median():+.2%}",
            "4 год": f"{df['ret_4h'].median():+.2%}", "12 год": f"{df['ret_12h'].median():+.2%}",
            "4 год, сер.": f"{df['ret_4h'].mean():+.2%}", "4 год > 0": f"{(df['ret_4h'] > 0).mean():.0%}",
            "4 год, ATR": f"{df['ret_4h_atr'].median():+.2f}", "стоп, % ціни": f"{df['R_pct'].median():.1%}",
            "+1R до стопу": f"{df['hit1R'].mean():.0%}", "+2R до стопу": f"{df['hit2R'].mean():.0%}",
            "R сер.": f"{df['R'].mean():+.3f}", "R без топ-1%": f"{df['R'].sort_values().iloc[:max(1, int(len(df) * 0.99))].mean():+.3f}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Прокол лоу з викупом / кульмінація + розворот дельти з футпрінту архіву Binance")
    ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    pd.set_option("display.width", 250)
    coins = sorted(os.listdir(CACHE))
    reviewed = {t.upper() for t in read_tickers_file("tickers.txt")}
    res: dict[str, list] = {}
    bars_all = {}
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        pumps = load_pumps(conn)
        for n, base in enumerate(coins, 1):
            x = load_coin(conn, base)
            if x is None:
                continue
            b = bar_metrics(*x)
            ev = events(b)
            for name, m in ev.items():
                if name == "к: усі бари":
                    m = m & (np.arange(len(m)) % 16 == 0)          # випадкова вибірка: кожен 16-й бар (раз на 4 год)
                res.setdefault(name, []).append(outcomes(b, m, name.startswith("U"), base))
            bars_all[base] = (b, ev)
            if n % 40 == 0:
                print(f"  {n}/{len(coins)}", flush=True)
    tab = {k: pd.concat(v, ignore_index=True) for k, v in res.items()}
    s = pd.DataFrame({k: summary(v) for k, v in tab.items()}).T
    print("\nВсі монети (поза навчанням і навчальні), 15m Binance:\n" + s.to_string(), flush=True)

    # A: дельта під лоу від'ємна / додатна; контекст «гаряча монета» (ціна за добу ≥ +10%)
    A = tab["A прокол з викупом"]
    hot = []
    for r in A.itertuples():
        b = bars_all[r.base][0]
        i = b.index.get_loc(r.ts)
        hot.append(i >= 96 and b["price"].iloc[i] / b["price"].iloc[i - 96] - 1 >= 0.10)
    A = A.assign(hot=hot)
    split = pd.DataFrame({"A · дельта під лоу < 0": summary(A[A["below_delta"] < 0]),
                          "A · дельта під лоу ≥ 0": summary(A[A["below_delta"] >= 0]),
                          "A · монета +10% за добу": summary(A[A["hot"]]), "A · решта": summary(A[~A["hot"]]),
                          "A · навчальні тікери": summary(A[A["base"].isin(reviewed | set(pumps["base_asset"]))]),
                          "A · поза навчанням": summary(A[~A["base"].isin(reviewed | set(pumps["base_asset"]))])}).T
    print("\nA за підгрупами:\n" + split.to_string(), flush=True)

    # розмітка: A за ≤ 3 год до старту імпульсу проти звичайного 3-год вікна; U у першу годину фальстарту / імпульсу
    lab = []
    for kind in ("pump", "false_start"):
        for p in pumps[pumps["pattern_type"] == kind].itertuples():
            if p.base_asset not in bars_all:
                continue
            b, ev = bars_all[p.base_asset]
            pre = (b.index > p.start_ts - pd.Timedelta(hours=3)) & (b.index <= p.start_ts + pd.Timedelta(minutes=15))
            first = (b.index > p.start_ts - pd.Timedelta(minutes=15)) & (b.index <= p.start_ts + pd.Timedelta(hours=1))
            if pre.sum() < 8:
                continue
            lab.append({"kind": kind, "A_before": bool(ev["A прокол з викупом"][pre].any()),
                        "B_before": bool(ev["B кульмінація + дельта"][pre].any()), "U_first": bool(ev["U прокол хаю (шорт)"][first].any())})
    lab = pd.DataFrame(lab)
    base_rate = {}
    for name in ("A прокол з викупом", "B кульмінація + дельта", "U прокол хаю (шорт)"):
        hits, wins = 0, 0
        for b, ev in bars_all.values():
            m = ev[name].astype(int).rolling(12 if name[0] != "U" else 5, min_periods=1).max()
            hits += int(m.iloc[::12].sum())
            wins += len(m.iloc[::12])
        base_rate[name] = hits / max(wins, 1)
    lab_tab = pd.DataFrame({
        "імпульси": {"A за 3 год до старту": lab[lab.kind == "pump"]["A_before"].mean(),
                     "B за 3 год до старту": lab[lab.kind == "pump"]["B_before"].mean(),
                     "U у першу годину": lab[lab.kind == "pump"]["U_first"].mean(), "n": (lab.kind == "pump").sum()},
        "фальстарти": {"A за 3 год до старту": lab[lab.kind == "false_start"]["A_before"].mean(),
                       "B за 3 год до старту": lab[lab.kind == "false_start"]["B_before"].mean(),
                       "U у першу годину": lab[lab.kind == "false_start"]["U_first"].mean(), "n": (lab.kind == "false_start").sum()},
        "будь-яке вікно": {"A за 3 год до старту": base_rate["A прокол з викупом"], "B за 3 год до старту": base_rate["B кульмінація + дельта"],
                           "U у першу годину": base_rate["U прокол хаю (шорт)"], "n": None}}).round(3)
    print("\nРозмітка (частка випадків, де патерн був):\n" + lab_tab.to_string(), flush=True)

    lines = [f"# Прокол лоу з викупом і кульмінація з футпрінту ({datetime.now(timezone.utc):%Y-%m-%d})", "",
             f"{len(bars_all)} монет, 15m Binance, архів угод (кеш flow_archive). Правила — у docstring скрипту.", "",
             "## Усі події", "", s.to_markdown(), "", "## A за підгрупами", "", split.to_markdown(), "",
             "## Розмітка", "", lab_tab.to_markdown(), ""]
    os.makedirs("reports", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    A.to_csv(os.path.join("reports", "spring_events_A.csv"), index=False)
    print(f"\nзвіт: {OUT}")


if __name__ == "__main__":
    main()
